import argparse
import logging
import torch
from facellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from facellava.conversation import conv_templates, SeparatorStyle
from facellava.model.builder import load_pretrained_model
from facellava.utils import disable_torch_init
from facellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Following is a list of prompts that works well for different tasks.
# =================== AGE ESTIMATION - Use with Images ==========================
# Guess the age of the individual shown in the picture; justify your answer.
# Determine the approximate age of the person depicted, providing your reasoning.
# What age would you assign to the person in this visual, and why?
# Infer the age of the person in the photograph, explaining your rationale.
# Based on the image, what is your best estimate of the person's age, and what leads you to that conclusion?
# Provide an age estimation for the person pictured, and explain your estimation.
# =================== EMOTION RECOGNITION - Use with Videos ==========================
# What feeling does the video evoke?
# Identify the emotion conveyed in the video.
# What emotion is shown in the video?
# Explain the emotional content of the video.
# Analyze the emotions depicted in the video.
# What's the emotional tone of the video?
# =================== FACIAL ATTRIBUTES - Use with Images ===================================
# Analyze the facial characteristics shown in the picture.
# Identify the visible features of the face in the image.
# Describe the person's face as seen in the photograph.
# Give a description of the facial features present.
# Detail the appearance of the face depicted.
# Characterize the facial structure in the provided image.
# Specify the visible facial components.
# =================== FACIAL ACTION UNITS - Use with Images ===================================
# Identify the active facial action units in the provided image.
# Specify which facial action units are present in the image.
# List the facial action units visible in the supplied picture.
# Indicate the facial action units shown in the given photograph.
# Detail the facial muscle movements depicted in the image.
# Which facial action units appear in this image?



def parse_args():
    parser = argparse.ArgumentParser(description="Run model with input file and prompt")

    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to pass to the model')

    return parser.parse_args()

def main_video(args):
    ## use this for emotion recognition task

    disable_torch_init()
    video = args.file_path
    inp = args.prompt  ## Input prompt for the model.
    model_path = args.model_path
    # model_base = 'lmsys/vicuna-7b-v1.5'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)

def main_image(args):
    # use this for age, facial attributes, and action units tasks

    disable_torch_init()
    image = args.file_path

    inp = args.prompt  ## Input prompt for the model.
    use_landmarks=False
    model_path = args.model_path
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    image_processor = processor['image']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    cur_images = [image]

    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_processor(cur_images, return_tensors='pt')['pixel_values']]
    landmarks = None

    print(f"{roles[1]}: {inp}")
    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            landmarks=landmarks)

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)

if __name__ == '__main__':
    args = parse_args()
    print("Prompt: ", args.prompt)
    print("Model Path: ", args.model_path)
    print("File Path: ", args.file_path)
    if args.file_path.endswith('.mp4') or args.file_path.endswith('.avi'):
        logging.info("You have provided a video file.")
        logging.info("Running video inference...")
        logging.warning("Video inference works well with emotion recognition tasks.")
        logging.warning("For other tasks, please use image inference.")
        main_video(args)
    elif args.file_path.endswith('.jpg') or args.file_path.endswith('.png'):
        logging.info("You have provided an image file.")
        logging.info("Running image inference...")
        logging.warning("Image inference works well with age estimation, facial attributes, and action units tasks.")
        logging.warning("For emotion recognition tasks, please use video inference.")
        main_image(args)
    else:
        print("Unsupported file format. Please provide a video or image file.")
