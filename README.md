<div align="center">
  <img src="./assets/readme_assets/facellava_logo.png" width="400">

  <h1>Facial Expression and Attribute Understanding through Instruction Tuning</h1>
  <h3>WACV 2026</h3>

  <p>
    <a href="https://arxiv.org/abs/2504.07198">
      <img src="https://img.shields.io/badge/arXiv-2504.07198-b31b1b.svg?logo=arxiv" alt="arXiv">
    </a>
    <a href="https://github.com/ihp-lab/Face-LLaVA">
      <img src="https://img.shields.io/badge/Github-FaceLLaVA-black?logo=github" alt="GitHub">
    </a>
    <a href="https://face-llava.github.io/">
      <img src="https://img.shields.io/badge/Website-facellava.github.io-purple?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjwhLS0gU3ZnIFZlY3RvciBJY29ucyA6IGh0dHA6Ly93d3cub25saW5ld2ViZm9udHMuY29tL2ljb24gLS0+DQo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDEuMS8vRU4iICJodHRwOi8vd3d3LnczLm9yZy9HcmFwaGljcy9TVkcvMS4xL0RURC9zdmcxMS5kdGQiPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IiB2aWV3Qm94PSIwIDAgMjU2IDI1NiIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgMjU2IDI1NiIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+DQo8bWV0YWRhdGE+IFN2ZyBWZWN0b3IgSWNvbnMgOiBodHRwOi8vd3d3Lm9ubGluZXdlYmZvbnRzLmNvbS9pY29uIDwvbWV0YWRhdGE+DQo8Zz48Zz48cGF0aCBmaWxsPSIjRkZGRkZGIiBkPSJNMTI4LDI0NmMtNjUuMiwwLTExOC01Mi44LTExOC0xMThDMTAsNjIuOCw2Mi44LDEwLDEyOCwxMGM2NS4yLDAsMTE4LDUyLjgsMTE4LDExOEMyNDYsMTkzLjIsMTkzLjIsMjQ2LDEyOCwyNDZMMTI4LDI0NnogTTIzMS4zLDEyOGgtNTljMCwxNS43LTEuMiwzMC42LTMuMyw0NC4yaDUxLjlDMjI3LjMsMTU4LjgsMjMxLjMsMTQzLjksMjMxLjMsMTI4TDIzMS4zLDEyOHogTTIxMi4yLDE4N2gtNDUuOWMtMy43LDE2LjktOC45LDMxLjItMTUuMSw0MS40QzE3Ni40LDIyMi42LDE5Ny44LDIwNy41LDIxMi4yLDE4N0wyMTIuMiwxODd6IE05OC45LDExMy4zaDU4LjFjLTAuNC0xMC41LTEuNC0yMC4zLTIuNi0yOS41aC01Mi45QzEwMC4zLDkzLDk5LjQsMTAyLjgsOTguOSwxMTMuM0w5OC45LDExMy4zeiBNOTguNSwxMjhjMCwxNS45LDEuMSwzMC44LDMsNDQuMmg1My4xYzEuOC0xMy41LDIuOS0yOC4zLDIuOS00NC4ySDk4LjVMOTguNSwxMjh6IE0xMjgsMjMxLjNjMTAsMCwxOC44LTE3LjYsMjQuMi00NC4zaC00OC4zQzEwOS4yLDIxMy43LDExOCwyMzEuMywxMjgsMjMxLjNMMTI4LDIzMS4zeiBNMTA0LjksMjI4LjRjLTYuMy0xMC4yLTExLjUtMjQuNS0xNS4xLTQxLjRINDMuOEM1OC4yLDIwNy41LDc5LjYsMjIyLjYsMTA0LjksMjI4LjRMMTA0LjksMjI4LjR6IE0yNC44LDEyOEwyNC44LDEyOEwyNC44LDEyOEwyNC44LDEyOEwyNC44LDEyOHogTTM1LjEsMTcyLjNIODdjLTIuMS0xMy43LTMuMy0yOC42LTMuMy00NC4yaC01OUMyNC44LDE0My45LDI4LjcsMTU4LjgsMzUuMSwxNzIuM0wzNS4xLDE3Mi4zeiBNODQuMSwxMTMuM2MwLjUtMTAuNCwxLjUtMjAuMiwyLjktMjkuNUgzNWMtNC40LDkuMi03LjQsMTkuMS05LDI5LjVIODQuMUw4NC4xLDExMy4zeiBNNDMuNiw2OWg0Ni4xYzMuNy0xNi45LDguOC0zMS4yLDE1LjEtNDEuNEM3OS42LDMzLjQsNTgsNDguNSw0My42LDY5TDQzLjYsNjl6IE0xMjgsMjQuN2MtMTAsMC0xOC43LDE3LjctMjQsNDQuM0gxNTJDMTQ2LjcsNDIuNCwxMzgsMjQuNywxMjgsMjQuN0wxMjgsMjQuN3ogTTE1MS4xLDI3LjZjNi4yLDEwLjIsMTEuNCwyNC42LDE1LjEsNDEuNGg0Ni4xQzE5OCw0OC41LDE3Ni40LDMzLjQsMTUxLjEsMjcuNkwxNTEuMSwyNy42eiBNMjIxLDgzLjhoLTUyLjFjMS40LDkuMywyLjUsMTkuMiwzLDI5LjVIMjMwQzIyOC41LDEwMi44LDIyNS40LDkyLjksMjIxLDgzLjhMMjIxLDgzLjh6Ii8+PC9nPjwvZz4NCjwvc3ZnPg==" alt="Website">
    </a>
    <a href="https://huggingface.co/chaubeyG/FaceLLaVA">
      <img src="https://img.shields.io/badge/Weights-FaceLLaVA-orange?logo=huggingface" alt="Model Weights">
    </a>
    <a href="LICENSE.rst">
      <img src="https://img.shields.io/badge/License-USC%20Research-green?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjwhLS0gU3ZnIFZlY3RvciBJY29ucyA6IGh0dHA6Ly93d3cub25saW5ld2ViZm9udHMuY29tL2ljb24gLS0+DQo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDEuMS8vRU4iICJodHRwOi8vd3d3LnczLm9yZy9HcmFwaGljcy9TVkcvMS4xL0RURC9zdmcxMS5kdGQiPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IiB2aWV3Qm94PSIwIDAgMjU2IDI1NiIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgMjU2IDI1NiIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+DQo8bWV0YWRhdGE+IFN2ZyBWZWN0b3IgSWNvbnMgOiBodHRwOi8vd3d3Lm9ubGluZXdlYmZvbnRzLmNvbS9pY29uIDwvbWV0YWRhdGE+DQo8Zz4gPHBhdGggZmlsbD0iI0ZGRkZGRiIgZD0iTTIyOC4zLDIyMi40SDI3LjdjLTkuOCwwLTE3LjctNy45LTE3LjctMTcuN1Y1MS4zYzAtOS44LDcuOS0xNy43LDE3LjctMTcuN2gyMDAuNiBjOS44LDAsMTcuNyw3LjksMTcuNywxNy43djE1My40QzI0NiwyMTQuNSwyMzguMSwyMjIuNCwyMjguMywyMjIuNHogTTI3LjcsNDUuNGMtMy4zLDAtNS45LDIuNi01LjksNS45bDAsMHYxNTMuNCBjMCwzLjMsMi42LDUuOSw1LjksNS45aDIwMC42YzMuMywwLDUuOS0yLjYsNS45LTUuOVY1MS4zYzAtMy4zLTIuNi01LjktNS45LTUuOUgyNy43eiIvPiA8cGF0aCBmaWxsPSIjRkZGRkZGIiBkPSJNMTIyLjEsODAuOEg1MS4zYy0zLjMsMC01LjktMi42LTUuOS01LjljMC0zLjMsMi42LTUuOSw1LjktNS45aDcwLjhjMy4zLDAsNS45LDIuNiw1LjksNS45IEMxMjgsNzguMiwxMjUuNCw4MC44LDEyMi4xLDgwLjh6IE0xMjIuMSwxMTYuMkg1MS4zYy0zLjMsMC01LjktMi42LTUuOS01LjlzMi42LTUuOSw1LjktNS45aDcwLjhjMy4zLDAsNS45LDIuNiw1LjksNS45IFMxMjUuNCwxMTYuMiwxMjIuMSwxMTYuMnogTTEyMi4xLDEzOS44SDUxLjNjLTMuMywwLTUuOS0yLjYtNS45LTUuOXMyLjYtNS45LDUuOS01LjloNzAuOGMzLjMsMCw1LjksMi42LDUuOSw1LjkgUzEyNS40LDEzOS44LDEyMi4xLDEzOS44eiBNMTIyLjEsMTYzLjRINTEuM2MtMy4zLDAtNS45LTIuNi01LjktNS45czIuNi01LjksNS45LTUuOWg3MC44YzMuMywwLDUuOSwyLjYsNS45LDUuOSBTMTI1LjQsMTYzLjQsMTIyLjEsMTYzLjR6IE0xMTAuMywxODdoLTU5Yy0zLjMsMC01LjktMi42LTUuOS01LjljMC0zLjMsMi42LTUuOSw1LjktNS45aDU5YzMuMywwLDUuOSwyLjYsNS45LDUuOSBDMTE2LjIsMTg0LjQsMTEzLjYsMTg3LDExMC4zLDE4N3ogTTIyMS43LDg3LjJsLTkuNi03TDIwOC41LDY5aC0xMS45bC05LjYtN2wtOS42LDdoLTExLjlsLTMuNywxMS4zbC05LjYsN2wzLjcsMTEuM2wtMy43LDExLjMgbDkuNiw3bDEuNiw0LjhjMCwwLjIsMCwwLjQsMCwwLjZ2NTljMCwzLjMsMi42LDUuOSw1LjksNS45YzEuNiwwLDMuMS0wLjYsNC4yLTEuN2wxMy41LTEzLjVsMTMuNSwxMy41YzEuNywxLjcsNC4yLDIuMiw2LjQsMS4zIGMyLjItMC45LDMuNi0zLjEsMy42LTUuNXYtNTljMC0wLjIsMC0wLjQsMC0wLjZsMS42LTQuOGw5LjYtN2wtMy43LTExLjNMMjIxLjcsODcuMkwyMjEuNyw4Ny4yeiBNMTY2LjEsOTEuN2w1LjgtNC4ybDIuMi02LjhoNy4xIGw1LjgtNC4ybDUuOCw0LjJoNy4xbDIuMiw2LjhsNS44LDQuMmwtMi4yLDYuOGwyLjIsNi44bC01LjgsNC4ybC0yLjIsNi44aC03LjFsLTUuOCw0LjJsLTUuOC00LjJoLTcuMWwtMi4yLTYuOGwtNS44LTQuMmwyLjItNi44IEwxNjYuMSw5MS43eiBNMTkxLjIsMTU5LjJjLTIuMy0yLjMtNi0yLjMtOC4zLDBsLTcuNiw3LjZWMTI4aDIuMmw5LjYsN2w5LjYtN2gyLjJ2MzguOEwxOTEuMiwxNTkuMkwxOTEuMiwxNTkuMnoiLz48L2c+DQo8L3N2Zz4=" alt="License">
    </a>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python" alt="Python Version">
    </a>
    
  </p>
  <br>
</div>

This is the official codebase of the **WACV 2026 Round 1** Early Accept paper (6.4% acceptance rate) - Face-LLaVA: Facial Expression and Attribute Understanding through Instruction Tuning. 

---

## 🧾 Abstract

The human face plays a central role in social communication, necessitating the use of performant computer vision tools for human-centered applications. We propose Face-LLaVA, a multimodal large language model for face-centered, in-context learning, including facial expression and attribute recognition. Additionally, Face-LLaVA is able to generate natural language descriptions that can be used for reasoning. Leveraging existing visual databases, we first developed FaceInstruct-1M, a face-centered database for instruction tuning MLLMs for face processing. We then developed a novel face-specific visual encoder powered by Face-Region Guided Cross-Attention that integrates face geometry with local visual features. We evaluated the proposed method across nine different datasets and five different face processing tasks, including facial expression recognition, action unit detection, facial attribute detection, age estimation and deepfake detection. Face-LLaVA achieves superior results compared to existing open-source MLLMs and competitive performance compared to commercial solutions. Our model output also receives a higher reasoning rating by GPT under a zero-shot setting across all the tasks. 

<div align="center">
  <img src="./assets/readme_assets/face_llava_main_figure.png" width="75%">
</div>

---

## 📣 News

- [Oct. 2025] Initial release of the official codebase and model weights. Stay tuned for more details and the dataset.
- [Sept. 2025] Face-LLaVA accepted in the first round of WACV 2026 (6.4% acceptance rate). See you in Tucson!

## 🏆 Results


<details>
<summary>Facial Expression Recognition</summary>

<div align="center">
  <img src="./assets/result_tables/results_fer.png" width="50%">
</div>

</details>

---

<details>
<summary>Facial Attribute and Age Estimation</summary>

<div align="center">
  <img src="./assets/result_tables/results_age_attr.png" width="50%">
</div>

</details>

## 📦 Repository Structure

```bash
├── cache_dir/             # Automatically created to store downloaded LanguageBind models
├── checkpoints/           # Folder to store model weights
├── facellava/             # Main source code
├── scripts/               # Training scripts for Face-LLaVA
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

1. Download the model weights from [huggingface](https://huggingface.co/chaubeyG/FaceLLaVA) inside `checkpoints/` folder so that the structure becomes - `./checkpoints/FaceLLaVA`.

2. Crop the input image/video using `tools/crop_face.py` before further processing. 

    Use the following command to crop an image

    ```python
    python crop_face.py \
        --mode image \
        --image_path "/path/to/input.jpg" \
        --output_image_path "/path/to/output_cropped.jpg"
    ```

    Use the following command to crop a video
    ```python
    python crop_face.py \
        --mode video \
        --video_path "/path/to/input/video.mp4" \
        --output_video_path "/path/to/output/cropped_video.mp4" \
        --temp_dir "/path/to/temp"
    ```

3. Run the following command for inference.

    ```bash
    CUDA_VISIBLE_DEVICES=0 python inference.py --model_path="./checkpoints/FaceLLaVA" \
    --file_path="./assets/demo_inputs/face_attr_example_1.png" --prompt="What are the facial attributes in the given image?"
    ```

4. **Currently the following face perception tasks are supported along with the best modality suited for that task - Emotion(Video), Age(Image), Facial Attributes(Image), Facial Action Units(Image)**

5. A list of prompts that work well for different tasks is present in `./assets/good_prompts`.

### ✅ Repository Progress

- [ ] Dataset Release
- [x] Training Script
- [x] Inference Code
- [x] Model Weights 

## ⚖️ License

This codebase is distributed under the USC Research license. See [LICENSE.rst](LICENSE.rst) for more details. This repo uses parts of code from the [VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) repo and our codebase inherit their license for those.

## 🙌 Credits

This codebase builds upon the following excellent works: [VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), [LLaVA](https://github.com/haotian-liu/LLaVA) and [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT). We gratefully acknowledge their contributions to the open-source community.

## 🪶 Citation

```latex
@article{chaubey2025face,
  title={Face-LLaVA: Facial Expression and Attribute Understanding through Instruction Tuning},
  author={Chaubey, Ashutosh and Guan, Xulang and Soleymani, Mohammad},
  journal={arXiv preprint arXiv:2504.07198},
  year={2025}
}
```
