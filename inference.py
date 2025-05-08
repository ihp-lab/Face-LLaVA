import torch
from facellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from facellava.conversation import conv_templates, SeparatorStyle
from facellava.model.builder import load_pretrained_model
from facellava.utils import disable_torch_init
from facellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def main_video():
    disable_torch_init()
    video = '/wekafs/ict/achaubey/emotion_reasoning/data/ferv39k/ferv39k_processed_new/face_cropped/Conflict/Happy/0001.mp4'
    inp = 'What facial action units are engaged in the image?'
    # model_path = 'LanguageBind/Video-LLaVA-7B'
    # model_path = "./checkpoints/facellava-7b-lora"
    model_path = "./checkpoints/facellava-7b-attn"
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

def main_image():
    disable_torch_init()
    # image = '/wekafs/ict/achaubey/emotion_reasoning/data/dfew/dfew_processed_new/frames_8/5660/5660_6.png'
    image = "/wekafs/ict/achaubey/emotion_reasoning/data/dfew/dfew_processed_new/frames_8/79/79_1.png"
    inp = 'What is the age of the person in the image?'
    use_landmarks=False
    # model_path = 'LanguageBind/Video-LLaVA-7B'
    # model_path = "./checkpoints/facellava-7b-lora"
    model_path = "/wekafs/ict/achaubey/emotion_reasoning/code/Video-LLaVA/checkpoints_final/videollava-7b-morphii_lr2e-5_8ep"
    # model_base = 'lmsys/vicuna-7b-v1.5'
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

    if not use_landmarks:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_processor(cur_images, return_tensors='pt')['pixel_values']]
        landmarks = None
    else:
        try:
            cur_proc_image = image_processor(cur_images, landmarks_2d = cur_landmarks, return_tensors='pt')
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in cur_proc_image['pixel_values']]
            landmarks = [lm.to(model.device, dtype=torch.float16) for lm in cur_proc_image['landmarks']]
        except:
            with open("disfa_errors.txt", "a") as f:
                f.write(f"{cur_images} \n")
        print("image_tensor", len(image_tensor), image_tensor[0].shape)
        print("landmarks", len(landmarks), landmarks[0].shape)

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
    main_image()
