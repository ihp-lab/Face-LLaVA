import torch
import torch.nn as nn
import re
import math
import torch.nn.functional as F


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class WeightMaskMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads = 16, d_in = None, use_layer_norm=False, use_residual=False):
        super(WeightMaskMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        if d_in is None:
            d_in = d_model

        if (d_in != d_model) and use_residual:
            self.res_linear_proj = nn.Sequential(
                nn.Linear(d_in, d_model)
            )

        if use_layer_norm:
            self.pre_norm = nn.LayerNorm(normalized_shape=d_in)
            self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model)

        self.W_q = nn.Linear(d_in, d_model)
        self.W_k = nn.Linear(d_in, d_model)
        self.W_v = nn.Linear(d_in, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.mask_avg_pool = nn.AvgPool2d(14, stride=14)
        
    def scaled_dot_product_attention(self, Q, K, V, weight_mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if weight_mask is not None:
            # print(weight_mask.shape)
            # print(attn_scores.shape)
            weight_mask = weight_mask.unsqueeze(1)
            weight_mask = weight_mask.expand(-1, 16, -1, -1) # expand to cover the num_heads dimension and replicate mask for all heads
            assert (attn_scores.shape == weight_mask.shape)
            # print("2 weight_mask.shape", weight_mask.shape)
            attn_scores = attn_scores * weight_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, x, weight_mask=None):
        # print(x.shape)
        B, T, N, D = x.shape
        x = x.view(-1, N, D)

        if self.use_layer_norm:
            x = self.pre_norm(x)

        if weight_mask is not None:
            # patch and avg pool mask
            batch_size, seq_length, _, _ = weight_mask.size()
            weight_mask = self.mask_avg_pool(weight_mask) # B, T, 16, 16
            weight_mask = weight_mask.view(batch_size, seq_length, 256)  # B, T, 256
            weight_mask = weight_mask.view(-1, 256) # B*T, 256
            cls_mask = torch.ones(batch_size*seq_length, 1) # we need to put 1 in the mask for the [cls] token for all samples
            cls_mask = cls_mask.to(weight_mask.device)
            weight_mask = torch.cat((cls_mask, weight_mask), dim=1) # B*T, 257
            weight_mask = weight_mask.unsqueeze(2) # B*T, 257, 1
            weight_mask_t = weight_mask.transpose(1, 2) # B*T, 1, 257
            weight_mask = torch.bmm(weight_mask, weight_mask_t).to(dtype=x.dtype)
            # print("1 weight_mask.shape", weight_mask.shape)

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn_output = self.scaled_dot_product_attention(Q, K, V, weight_mask)
        attn_output = self.combine_heads(attn_output)

        if self.use_residual:
            if x.shape[-1] != attn_output.shape[-1]:
                proj_x = self.res_linear_proj(x)
                attn_output = attn_output + proj_x

        if self.use_layer_norm:
            attn_output = self.layer_norm1(attn_output)

        output = self.W_o(attn_output)

        if self.use_residual:
            output = output + attn_output

        return output.view(B, T, N, -1)

class WeightMaskProjector(nn.Module):

    def __init__(self, mm_hidden_size, hidden_size, num_linear = 2, num_heads = 16):
        super(WeightMaskProjector, self).__init__()

        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, num_linear):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.linear_proj = nn.Sequential(*modules)

        self.weight_attn = WeightMaskMultiHeadAttention(hidden_size, num_heads)

    def forward(self, x, weight_mask=None):
        x = self.linear_proj(x)
        output = self.weight_attn(x, weight_mask)
        return output

class WeightMaskProjectorSimple(nn.Module):

    def __init__(self, mm_hidden_size, hidden_size, num_linear = 2, num_heads = 16, use_layer_norm=False, use_residual=False):
        super(WeightMaskProjectorSimple, self).__init__()

        self.weight_attn = WeightMaskMultiHeadAttention(hidden_size, num_heads, d_in = mm_hidden_size,
                                                        use_layer_norm=use_layer_norm, use_residual=use_residual)

    def forward(self, x, weight_mask=None):
        output = self.weight_attn(x, weight_mask)
        return output

class LandmarkRegionProjector(nn.Module):
    """
    Takes a tensor of shape (B, 8, 136) containing 8 frames of 68 (x,y) landmarks,
    splits the 68 landmarks into 9 regions, and for each region
    projects the flattened coordinates into a out_dim-dim embedding.

    Output shape: (B, 8, 9, out_dim)
    """
    def __init__(self, out_dim):
        super().__init__()
        self.facial_regions = {
            'face': slice(0, 17),
            'eyebrow1': slice(17, 22),
            'eyebrow2': slice(22, 27),
            'nose': slice(27, 31),
            'nostril': slice(31, 36),
            'eye1': slice(36, 42),
            'eye2': slice(42, 48),
            'lips': slice(48, 60),
            'teeth': slice(60, 68)
        }
        
        # We create one linear layer per facial region,
        # sized according to (#points_in_region * 2) --> 4096
        self.region_layers = nn.ModuleDict()
        for region_name, region_slice in self.facial_regions.items():
            num_points = region_slice.stop - region_slice.start  # e.g. 17 for 'face'
            in_dim = num_points * 2                               # times 2 for (x,y)
            
            # single-layer perceptron (linear)
            self.region_layers[region_name] = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        x: (B, T, 136) for videos or (B, 136) for images
        Returns:
            (B, T, 9, out_dim) for videos or (B, 9, out_dim) for images
        """
        # Handle input with or without time dimension
        if x.ndim == 2:  # Case: images (B, 136)
            x = x.unsqueeze(1)  # Add a temporal dimension => (B, 1, 136)
            is_image = True
        else:  # Case: videos (B, T, 136)
            is_image = False

        B, F, _ = x.shape  # F=8 frames
        # Reshape to (B, 8, 68, 2)
        x_points = x.view(B, F, 68, 2)
        
        region_embeddings = []
        
        # Iterate over each region in the dictionary
        for region_name, region_slice in self.facial_regions.items():
            # slice out the relevant landmark points => shape (B, F, num_points, 2)
            region_coords = x_points[:, :, region_slice, :]  
            # Flatten the last dimension => shape (B, F, num_points*2)
            region_coords_flat = region_coords.flatten(start_dim=2)  
            
            # Pass through the region-specific linear layer => (B, F, out_dim)
            projected = self.region_layers[region_name](region_coords_flat)
            
            # Add an extra dimension for "region" so we can concat
            # => shape (B, F, 1, out_dim)
            projected = projected.unsqueeze(2) 
            region_embeddings.append(projected)
        
        # Concatenate over the region dimension => (B, F, 9, 4096)
        out = torch.cat(region_embeddings, dim=2)

        # Remove the time dimension if the input was an image
        if is_image:
            out = out.squeeze(1)  # (B, 9, out_dim)

        return out

class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)  # LayerNorm after residual

    def forward(self, img_feats, landmark_feats):
        """
        img_feats: (B, T, N, D)
        landmark_feats: (B, T, L, D)

        B: Batch size
        T: Could be 'time' or something else, dimension=8 in your example
        N: Number of image tokens (e.g., 257 for ViT)
        L: Number of landmark tokens
        D: hidden_size (embedding dimension)
        """

        B, T, N, D = img_feats.shape
        # Flatten the first two dims so MultiheadAttention sees (batch_size, seq_len, dim).
        # We'll call it "BT" = B*T
        img_feats_squished = img_feats.view(-1, N, D)            # shape: (B*T, N, D)
        landmark_feats_squished = landmark_feats.view(-1, landmark_feats.shape[2], D)  # (B*T, L, D)

        # Cross-attention:
        # Query = img_feats_squished
        # Key   = landmark_feats_squished
        # Value = landmark_feats_squished
        output, _ = self.attn(
            query=img_feats_squished, 
            key=landmark_feats_squished, 
            value=landmark_feats_squished
        )
        # Residual connection
        output = img_feats_squished + output  # shape: (B*T, N, D)

        # LayerNorm
        output = self.ln(output)  # shape: (B*T, N, D)

        # Reshape back to (B, T, N, D)
        output = output.view(B, T, N, D)
        return output


class LandmarkGuidedCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.cross_attn = CrossAttentionModule(hidden_size, num_heads)
        self.linear_proj = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)  # Another LayerNorm if desired

    def forward(self, img_feats, landmark_feats, landmarks=None, branch_scale = 0.001):
        # Handle input with or without time dimension
        if img_feats.ndim == 3:  # Case: images 
            img_feats = img_feats.unsqueeze(1)  # Add a temporal dimension
            landmark_feats = landmark_feats.unsqueeze(1)
            is_image = True
        else:  # Case: videos 
            is_image = False

        # 1) Cross-attention step
        x = self.cross_attn(img_feats, landmark_feats)  # shape: (B, T, N, D)

        # 2) Linear projection
        x = self.linear_proj(x)                         # shape: (B, T, N, D)

        x = branch_scale * x + img_feats

        # 3) Optional LayerNorm after linear
        # x = self.ln(x)
        # Remove the time dimension if the input was an image
        if is_image:
            x = x.squeeze(1)  # (B, 257, out_dim)
        return x

class CrossAttentionModuleWMask(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)  # LayerNorm after residual connection

    def forward(self, img_feats, landmark_feats, mask=None):
        """
        img_feats: (B, T, N, D)
        landmark_feats: (B, T, L, D)
        mask: (B, T, N, L), optional attention mask based on distances

        B: Batch size
        T: Temporal dimension or sequence length
        N: Number of image tokens (e.g., 257 for ViT)
        L: Number of landmark tokens (e.g., 9 combined regions)
        D: hidden_size (embedding dimension)
        """
        B, T, N, D = img_feats.shape
        _, _, L, _ = landmark_feats.shape

        # Project query, key, value
        query = self.q_proj(img_feats).view(B * T, N, self.num_heads, D // self.num_heads).transpose(1, 2)  # (B*T, H, N, D/H)
        key = self.k_proj(landmark_feats).view(B * T, L, self.num_heads, D // self.num_heads).transpose(1, 2)  # (B*T, H, L, D/H)
        value = self.v_proj(landmark_feats).view(B * T, L, self.num_heads, D // self.num_heads).transpose(1, 2)  # (B*T, H, L, D/H)

        # Compute raw attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D // self.num_heads)  # (B*T, H, N, L)

        # Apply mask if provided
        if mask is not None:
            mask = mask.view(B * T, 1, N, L)  # Broadcast over heads (B*T, 1, N, L)
            attn_scores += mask  # Add mask directly to attention scores before softmax
            print("Mask added to the attention scores...")

        # Normalize attention scores
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B*T, H, N, L)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, value)  # (B*T, H, N, D/H)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B * T, N, D)  # (B*T, N, D)

        # Residual connection and LayerNorm
        output = self.ln(img_feats.view(B * T, N, D) + attn_output)  # (B*T, N, D)

        # Reshape back to original dimensions
        return output.view(B, T, N, D)  # (B, T, N, D)

class LandmarkGuidedCrossAttentionWMask(nn.Module):
    def __init__(self, hidden_size, num_heads=8, grid_size=16):
        super().__init__()
        self.cross_attn = CrossAttentionModuleWMask(hidden_size, num_heads)
        self.linear_proj = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)  # Optional LayerNorm after linear projection
        self.grid_size = grid_size

        # Define landmark regions
        self.facial_regions = {
            'face': slice(0, 17),
            'eyebrow1': slice(17, 22),
            'eyebrow2': slice(22, 27),
            'nose': slice(27, 31),
            'nostril': slice(31, 36),
            'eye1': slice(36, 42),
            'eye2': slice(42, 48),
            'lips': slice(48, 60),
            'teeth': slice(60, 68)
        }

        # Generate grid coordinates for patch centroids (normalized 0-1)
        patch_size = 1 / (self.grid_size)
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(patch_size / 2, 1 - patch_size / 2, self.grid_size),
            torch.linspace(patch_size / 2, 1 - patch_size / 2, self.grid_size),
            indexing="ij"
        )
        self.grid_coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # Shape: (N-1, 2) for spatial tokens

        # Add dummy row for the cls token to make it (N, 2)
        cls_token_row = torch.zeros(1, 2)  # Placeholder for cls token
        self.grid_coords = torch.cat([cls_token_row, self.grid_coords], dim=0)  # Shape: (N, 2)

        # Reshape for broadcasting
        self.grid_coords = self.grid_coords.unsqueeze(0).unsqueeze(0).unsqueeze(3)  # Shape: (1, 1, N, 1, 2)

    def compute_distance_mask(self, img_feats, landmarks):
        """
        Compute distance-based attention mask between image tokens and landmarks.
        Args:
            img_feats: (B, T, N, D)
            landmarks: (B, T, 68, 2) normalized landmark coordinates (0-1).
        Returns:
            mask: (B, T, N, 9) distance-based mask for 9 landmark regions.
        """
        B, T, N, D = img_feats.shape
        landmarks = landmarks.view(B, T, 68, 2)
        print("landmarks", landmarks.shape)

        # Combine landmarks into 9 regions
        grouped_landmarks = torch.zeros(B, T, 9, 2, device=landmarks.device)  # Shape: (B, T, 9, 2)
        for i, indices in enumerate(self.facial_regions.values()):
            cur_lm_centroid = landmarks[:, :, indices].mean(dim=2)  # Compute centroid per region
            grouped_landmarks[:, :, i] = cur_lm_centroid

        # Reshape for broadcasting
        grouped_landmarks = grouped_landmarks.unsqueeze(2)  # Shape: (B, T, 1, 9, 2)

        # same device
        self.grid_coords = self.grid_coords.to(device=landmarks.device)
        print("self.grid_coords", self.grid_coords.shape)
        print("grouped_landmarks", grouped_landmarks.shape)

        # Compute Euclidean distances
        distances = torch.norm(self.grid_coords - grouped_landmarks, dim=-1)  # Shape: (B, T, N, 9)
        print("distances", distances.shape)

        # For the cls token (first token), set all distances to zero
        distances[:, :, 0, :] = 0  # Set cls token distances to zero

        # Convert distances to attention mask (closer -> higher score)
        mask = torch.softmax(-distances, dim=-1)  # Negative sign to prioritize closer distances
        print("mask", distances.shape)
        return mask


    def forward(self, img_feats, landmark_feats, landmarks, branch_scale=0.001):
        """
        img_feats: (B, T, N, D)
        landmark_feats: (B, T, 9, D)
        landmarks: (B, T, 68, 2) normalized landmark coordinates
        branch_scale: Scaling factor for the residual branch
        """
        # Compute distance-based mask
        mask = self.compute_distance_mask(img_feats, landmarks)  # Shape: (B, T, N, 9)

        # Cross-attention step with mask
        x = self.cross_attn(img_feats, landmark_feats, mask)  # Shape: (B, T, N, D)

        # Linear projection
        x = self.linear_proj(x)  # Shape: (B, T, N, D)

        # Scaled residual connection
        x = branch_scale * x + img_feats  # Shape: (B, T, N, D)

        # Optional LayerNorm (uncomment if desired)
        # x = self.ln(x)

        return x



def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "naive_face_mask_adapter":
        return WeightMaskProjector(config.mm_hidden_size, config.hidden_size)
    
    if projector_type == "naive_face_mask_adapter_simple":
        return WeightMaskProjectorSimple(config.mm_hidden_size, config.hidden_size)

    if projector_type == "naive_face_mask_adapter_simple_w_resnln":
        return WeightMaskProjectorSimple(config.mm_hidden_size, config.hidden_size, 
                                        use_layer_norm=True, use_residual=True)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "mlp2x_gelu_w_mask_wt":
        mlp_depth = 2
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_landmark_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'landmarks_projector_type', 'linear')

    # print("LANDMARKS_PROJECTOR_TYPE ===>", projector_type)

    if projector_type == 'linear':
        return nn.Linear(config.landmarks_hidden_size, config.hidden_size)

    if projector_type == "landmarks_region_projector":
        return LandmarkRegionProjector(config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.landmarks_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_global_landmark_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'global_landmarks_projector_type', 'linear')

    # print("GLOBAL_LANDMARKS_PROJECTOR_TYPE ===>", projector_type)

    if projector_type == 'linear':
        return nn.Linear(config.landmarks_hidden_size, config.hidden_size)

    if projector_type == "landmarks_region_projector":
        return LandmarkRegionProjector(config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.landmarks_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_landmark_cross_attn(config, delay_load=False, **kwards):
    attn_type = getattr(config, 'cross_attn_type', 'simple')
    # print("ATTENTION_TYPE ===>", attn_type)
    if attn_type == "simple":
        return LandmarkGuidedCrossAttention(config.hidden_size)
    elif attn_type == "masked":
        return LandmarkGuidedCrossAttentionWMask(config.hidden_size)
    raise ValueError(f'Unknown cross attention type: {attn_type}')

