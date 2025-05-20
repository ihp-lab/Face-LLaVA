# Face-LLaVA: Facial Expression and Attribute Understanding through Instruction Tuning

[![arXiv](https://img.shields.io/badge/arXiv-2305.00000v1-b31b1b.svg)](https://arxiv.org/abs/2504.07198)
[![Model Weights](https://img.shields.io/badge/Download-Model%20Weights-green)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

This is the official codebase of the paper - Face-LLaVA: Facial Expression and Attribute Understanding through Instruction Tuning. 

---

## 🧾 Abstract

The human face plays a central role in social communication, necessitating the use of performant computer vision tools for human-centered applications. We propose Face-LLaVA, a multimodal large language model for face-centered, in-context learning, including facial expression and attribute recognition. Additionally, Face-LLaVA is able to generate natural language descriptions that can be used for reasoning. Leveraging existing visual databases, we first developed FaceInstruct-1M, a face-centered database for instruction tuning MLLMs for face processing. We then developed a novel face-specific visual encoder powered by Face-Region Guided Cross-Attention that integrates face geometry with local visual features. We evaluated the proposed method across nine different datasets and five different face processing tasks, including facial expression recognition, action unit detection, facial attribute detection, age estimation and deepfake detection. Face-LLaVA achieves superior results compared to existing open-source MLLMs and competitive performance compared to commercial solutions. Our model output also receives a higher reasoning rating by GPT under a zero-shot setting across all the tasks. Both our dataset and model wil be released at this https URL to support future advancements in social AI and foundational vision-language research.

---

## 📦 Repository Structure

```bash
├── cache_dir/             # will automatically be created to download LanguageBind image and video models from huggingface
├── checkpoints/           # create a new folder by this name
├── facellava/             # Main source code
├── scripts/               # Training scripts for FaceLLaVA

```

---

## 🔧 Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/ac-alpha/face-llava.git
    cd Face-LLaVA
    ```

2. **Create a virtual environment** (recommended)
    ```bash
    conda create -n facellava python=3.10 -y
    conda activate facellava
    ```

3. **Install torch**
    ```bash
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

    <details>
    <summary>Potential issues</summary>

    - You might want to download PyTorch for a different version of CUDA. We download it for CUDA-12.1 but we have tested it on a machine with CUDA-12.2 as well. However, you might need to change this depending on your machine.
    - Based on the above, you might also have to upgrade/downgrade torch. 
    
    </details>
    

4. **Install in editable mode for development**:
    ```bash
    pip install -e .
    pip install -e ".[train]" ## if you want to train your own model
    ```

5. **Install other libraries**:
    ```bash
    pip install flash-attn --no-build-isolation ## recommended but not required
    pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
    ```


---

## 🎯 Inference

1. Download the model weights from [here]() and unzip them inside a `checkpoints/` folder so that the structure becomes - `./checkpoints/facellava-7b-wolm`.

2. ***Make sure that the input video or image is already face-cropped as the current version does not support automatic cropping.***

3. Run the following command for inference.

    ```bash
    CUDA_VISIBLE_DEVICES=0 python inference.py --model_path="./checkpoints/facellava-7b-wolm" --file_path="your_file_path_to_png_or_mp4" --prompt="What are the facial attributes in the given image."
    ```

4. Currently the following face perception tasks are supported along with the best modality suited for that task - Emotion(Video), Age(Image), Facial Attributes(Image), Facial Action Units(Image)

5. A list of prompts that work well for different tasks is present in `./assets/good_prompts`.

### ✅ Repository Progress

- [ ] Training Script
- [ ] Evaluation Metrics
- [ ] Dataset Release & Preprocessing Code
- [ ] Inference Code (with Landmarks & Auto Face Cropping)
- [x] Inference Code (Basic)
- [x] Model Weights (w/o Landmarks)