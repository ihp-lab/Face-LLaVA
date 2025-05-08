import torch
from PIL import Image
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
import numpy as np

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_image_transform(config):
    config = config.vision_config
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image
        ]
    )
    return transform


def load_and_transform_image(image_path, transform, landmark_2d_path=None):
    image = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path
    image_outputs = transform(image)
    landmark_outputs=None
    if landmark_2d_path is not None:
        landmark = np.load(landmark_2d_path)
        landmark_outputs = torch.from_numpy(landmark)
        landmark_outputs = landmark_outputs / 256.  # normalize the 2d coordinates
        landmark_outputs = landmark_outputs.view(-1)
        assert(landmark_outputs.shape[0] == 136)
        # print("landmark inside image read and transform", landmark_outputs.shape)
    return image_outputs, landmark_outputs

class LanguageBindImageProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindImageTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_image_transform(config)
        self.image_processor = load_and_transform_image
        self.tokenizer = tokenizer
        self.image_mean = OPENAI_DATASET_MEAN
        self.crop_size = {'height': 224, 'width': 224}

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, landmarks_2d=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            if landmarks_2d is not None:
                landmarks_2d = make_list_of_images(landmarks_2d)
                image_features = []
                landmark_features = []
                for image, lm in zip(images, landmarks_2d):
                    cur_img_feat, cur_lm_feat = self.image_processor(image, self.transform, landmark_2d_path=lm)
                    image_features.append(cur_img_feat)
                    landmark_features.append(cur_lm_feat)
                image_features = torch.stack(image_features)
                landmark_features = torch.stack(landmark_features)
            else:
                image_features = [self.image_processor(image, self.transform)[0] for image in images]
                image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            if landmarks_2d is not None:
                encoding["landmarks"] = landmark_features
            return encoding
        elif text is not None:
            return encoding
        else:
            if landmarks_2d is not None:
                return {"pixel_values": image_features, "landmarks": landmark_features}
            else:
                return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors, landmarks_2d=None):
        return self.__call__(images=images, return_tensors=return_tensors, landmarks_2d=landmarks_2d)

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
