
import torch
import cv2
import decord
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor, Grayscale
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

decord.bridge.set_bridge('torch')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_video_transform(config):
    config = config.vision_config
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    # RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif config.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform

def get_mask_transform(config):
    config = config.vision_config
    if config.video_decode_backend == 'pytorchvideo':
        raise NotImplementedError

    elif config.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif config.video_decode_backend == 'opencv':
        raise NotImplementedError
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
        video_path,
        transform,
        mask_path=None,
        mask_transform=None,
        landmark_2d_path=None,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
):

    # print("video_decode_backend", video_decode_backend)
    # print("video_decode_backend", video_decode_backend)
    # print("video_decode_backend", video_decode_backend)
    # print("video_decode_backend", video_decode_backend)

    if video_decode_backend != 'decord' and mask_path is not None:
        raise NotImplementedError

    mask_outputs = None
    landmark_outputs=None
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        # print("VIDEO PATH: ", video_path)
        # print("MASK PATH: ", mask_path)
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

        if mask_path is not None:
            decord_vr_mask = VideoReader(mask_path, ctx=cpu(0))
            duration_mask = len(decord_vr_mask)
            assert (duration_mask == duration)
            mask_data = decord_vr_mask.get_batch(frame_id_list)
            mask_data = mask_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            # assert (torch.equal(mask_data[0], mask_data[1]))
            mask_outputs = mask_transform(mask_data[:1])

        if landmark_2d_path is not None:
            landmark = np.load(landmark_2d_path)
            duration_lm = len(landmark)
            if duration_lm != duration:
                print("LM shape is not the same as the video frames... ERRORRRRR!!!!")
            assert (duration_lm == duration)
            landmark_outputs = torch.from_numpy(landmark[frame_id_list, :, :])
            landmark_outputs = landmark_outputs / 256.  # normalize the 2d coordinates
            landmark_outputs = landmark_outputs.view(num_frames, -1)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            _, frame = cv2_vr.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs, mask_outputs, landmark_outputs

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_video_transform(config)
        self.mask_transform = get_mask_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, masks=None, landmarks_2d=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            if masks is not None:
                masks = make_list_of_images(masks)
                assert len(masks) == len(images)
                image_features = []
                mask_features = []
                for image, mask in zip(images, masks):
                    cur_img_feat, cur_mask_feat, _ = self.image_processor(image, self.transform, mask_path=mask, mask_transform=self.mask_transform,
                                            video_decode_backend=self.config.vision_config.video_decode_backend,
                                            num_frames=self.config.vision_config.num_frames)
                    image_features.append(cur_img_feat)
                    mask_features.append(cur_mask_feat)
                image_features = torch.stack(image_features)
                mask_features = torch.stack(mask_features)
            elif landmarks_2d is not None:
                landmarks_2d = make_list_of_images(landmarks_2d)
                assert len(landmarks_2d) == len(images)
                image_features = []
                landmark_features = []
                for image, lm in zip(images, landmarks_2d):
                    cur_img_feat, _, cur_lm_feat = self.image_processor(image, self.transform, landmark_2d_path=lm,
                                            video_decode_backend=self.config.vision_config.video_decode_backend,
                                            num_frames=self.config.vision_config.num_frames)
                    image_features.append(cur_img_feat)
                    landmark_features.append(cur_lm_feat)
                image_features = torch.stack(image_features)
                landmark_features = torch.stack(landmark_features)
            else:
                image_features = [self.image_processor(image, self.transform,
                                                    video_decode_backend=self.config.vision_config.video_decode_backend,
                                                    num_frames=self.config.vision_config.num_frames)[0] for image in images]
                image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            if masks is not None:
                encoding["face_masks"] = mask_features
            if landmarks_2d is not None:
                encoding["landmarks"] = landmark_features
            return encoding
        elif text is not None:
            return encoding
        else:
            if masks is not None:
                return {"pixel_values": image_features, "face_masks": mask_features}
            if landmarks_2d is not None:
                return {"pixel_values": image_features, "landmarks": landmark_features}
            else:
                return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
