# VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction

**VLM-3R is a unified Vision-Language Model (VLM) framework integrating 3D reconstructive instruction tuning for deep spatial understanding from monocular video.**

The rapid advancement of Large Multimodal Models (LMMs) for 2D images and videos has motivated extending these models to understand 3D scenes, aiming for human-like visual-spatial intelligence. VLM-3R processes monocular video frames by employing a geometry encoder to derive implicit 3D tokens that represent spatial understanding. Through the utilization of Spatial-Visual–View Fusion technique and over 200K curated 3D reconstructive instruction tuning question-answer (QA) pairs, VLM-3R effectively aligns real-world spatial context with language instructions. This enables the model to perform monocular 3D spatial assistance and embodied reasoning.

[**Paper (arXiv)**](https://arxiv.org/abs/2505.20279) **|** [**Project Page**](https://vlm-3r.github.io/) **|** [**Code (GitHub)**](https://github.com/VITA-Group/VLM-3R) **|** [**Dataset (HF)**](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA) **|** [**VSTiBench (HF)**](https://huggingface.co/datasets/Journey9ni/vstibench)

## 🧑‍💻 Authors

[Zhiwen Fan](https://zhiwenfan.github.io/)<sup>1&dagger;\*</sup>, [Jian Zhang](https://jian-zhang-3dv.github.io/Jian-Zhang-3DV/)<sup>2\*</sup>, [Renjie Li](https://shadowiterator.github.io/)<sup>3</sup>, [Junge Zhang](https://andy-zd.github.io/)<sup>4</sup>, [Runjin Chen](https://chenrunjin.github.io/)<sup>1</sup>, [Hezhen Hu](https://alexhu.top/)<sup>1</sup>, [Kevin Wang](https://www.kevin-ai.com/)<sup>1</sup>, [Huaizhi Qu](https://sites.google.com/view/qhz991029)<sup>5</sup>, [Dilin Wang](https://wdilin.github.io/)<sup>6</sup>, [Zhicheng Yan](https://sites.google.com/view/zhicheng-yan)<sup>6</sup>, [Hongyu Xu](https://hyxu2006.github.io/)<sup>6</sup>, [Justin Theiss](https://www.linkedin.com/in/justin-d-theiss)<sup>6</sup>, [Tianlong Chen](https://tianlong-chen.github.io/)<sup>5</sup>, [Jiachen Li](https://jiachenli94.github.io/)<sup>4</sup>, [Zhengzhong Tu](https://vztu.github.io/)<sup>3</sup>, [Zhangyang Wang](https://vita-group.github.io/research.html)<sup>1</sup>, [Rakesh Ranjan](https://www.linkedin.com/in/rakesh-r-3848538)<sup>6</sup>

¹UT Austin   ²XMU   ³TAMU   ⁴UCR   ⁵UNC   ⁶Meta

†Corresponding Author. \*Equal contribution.

(zhiwenfan@utexas.edu)

## 📰 News

- **2025-06-11:** We have released the training/evaluation scripts and all associated data.
  - The main instruction tuning dataset, which includes training data for VSiBench and VSTiBench, is available on Hugging Face at [Journey9ni/VLM-3R-DATA](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA).
  - The test set for VSTiBench can be found at [Journey9ni/vstibench](https://huggingface.co/datasets/Journey9ni/vstibench).
- **2025-06-06:** VLM-3R data processing pipeline (including for VSiBench & VSTiBench) released.
  - **Note:** The data generation code for the `route plan` task in VSiBench is still being organized and is not yet open-sourced.
- **2025-06-03:** VSiBench evaluation code released.
- **2025-05-27:** Inference code and model weights released.

## Overview
![VLM-3R Project Overview](docs/images/teaser_00.jpg)

## 🚀 Key Innovations

- **End-to-End Monocular Video 3D Understanding:** VLM-3R directly processes monocular RGB videos without needing external depth sensors or pre-built 3D maps, significantly enhancing scalability and practical applicability.
- **3D Reconstructive Instruction Tuning:** Instruction tuning with over 200K QA pairs enables the model to effectively align visual information with 3D spatial context and language instructions.
- **Spatial-Visual-View Fusion:** A novel fusion mechanism integrates 3D geometric tokens, per-view camera tokens, and 2D appearance features for joint spatio-linguistic understanding.
- **Vision-Spatial-Temporal Intelligence Benchmark (VSTI-Bench):** A new benchmark with over 138.6K QA pairs, specifically designed to evaluate the model's understanding of spatio-temporal relationships evolving from camera motion within 3D environments.

## 🛠️ VLM-3R Architecture

The core of VLM-3R is a pre-trained Large Multimodal Model (LMM), integrated with modules for deriving geometric encodings, camera view encodings, and visual features from the input video; these diverse inputs are subsequently fused effectively with language representations. VLM-3R does not rely on pre-built 3D maps or external depth sensors. This design directly addresses key limitations of existing approaches, such as the common inadequacy of Video LLMs in perceiving rich spatial context from monocular video and the restrictive dependency of many specialized 3D-LLMs on prior 3D map or depth sensor inputs.

**Architecture Overview Diagram:**

[Video of VLM3R Network Architecture Demonstration](https://github.com/user-attachments/assets/f82f7905-879f-414a-a690-99fc471f2a50)

*Our method takes monocular video and language instruction as input. Visual Encoder coupled with Spatial Encoder extract frame-level appearance, camera view position, and globally aligned geometry. Visual-Geometry Fusion integrates these through attention and projection layers to create 3D-aware visual features for the LMM. During the inference stage, this fusion enables reliable spatial and temporal reasoning.*

**Key Components:**

- **3D Reconstructive Tokenization:** Utilizes the pre-trained CUT3R model to process monocular video frame-by-frame, extracting implicit latent representations (enriched feature tokens and camera view tokens). These tokens serve as rich 3D reconstructive tokens, compactly encoding observed 3D geometry and camera perspective without relying on explicit point clouds.

- **Spatial-Visual-View Fusion:** Employs a cross-attention mechanism where the VLM's native visual tokens (Hv) attend to a unified 3D representation (Z3D, formed by concatenated 3D feature tokens Ft′ and camera view tokens zt′). The output of this attention stage (Hattn) is then residually connected with the original visual tokens (Hv′=Hv+Hattn). This enriched representation Hv′ subsequently passes through a two-layer MLP projector for alignment with the LMM.

  ```
  Z_3D = Concat(F'_t, z'_t)
  H_attn = CrossAttention(Query: H_v, KeyValue: Z_3D)
  H'_v = H_v + H_attn
  ProjectedFeatures = MLP_2-layer(H'_v)
  ```

- **Training Objective & Fine-tuning Strategy:** Adopts the same learning objective as LLaVA-NeXT-Video. To achieve efficient adaptation, Low-Rank Adaptation (LoRA) is employed for fine-tuning, which involves updating parameters within the 3D fusion attention block and the projection layers.

## 📊 Datasets & Benchmarks

- **Instruction Tuning & Benchmark Training Data:** Our main instruction tuning dataset is publicly available on Hugging Face. This dataset also includes the training data for VSiBench and VSTiBench: [Journey9ni/VLM-3R-DATA](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA).
- **Data Generation Scripts:** The scripts for generating our instruction tuning data are now available. Please refer to the [`vlm_3r_data_process/README.md`](vlm_3r_data_process/README.md) for detailed instructions.
- **Multimodal Spatial Instruction Data Generation:** A scalable, automated data generation pipeline produced over **200,000** general question-answer pairs for spatial reasoning from monocular video, and **4,225** embodied route planning data instances generated using simulators. This data is derived from existing 3D datasets like ScanNet, ScanNet++, and ARKitScenes, processed via detailed spatio-temporal scene graphs to automatically generate QA pairs for tasks such as object counting, relative distance/direction, appearance order, object size, absolute distance, and room size.
- **Vision-Spatial-Temporal Intelligence Benchmark (VSTI-Bench):** Contains approximately **138,600** QA pairs to assess LMMs' ability to perceive and reason about dynamic spatial configurations. The VSTiBench **test set** is available on [Hugging Face](https://huggingface.co/datasets/Journey9ni/vstibench).

## ⚙️ Setup

### 1. Clone Repository and Submodules

```
git clone https://github.com/VITA-Group/VLM-3R.git
cd VLM-3R
git submodule update --init --recursive
```

### 2. Environment Setup

1. **Create conda environment:**

   ```
   conda create -n vlm3r python=3.10 -y
   conda activate vlm3r
   ```

2. **Install base packages:**

   ```
   pip install --upgrade pip
   conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   ```

3. **Install project dependencies:**

   ```
   pip install -e ".[train]"
   # Note: The FlashAttention wheel URL might be specific. Consider verifying compatibility.
   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   pip install decord openai accelerate==0.29.1
   ```

### 3. Install CUT3R

1. **Install requirements:**

   ```
   cd CUT3R
   pip install -r requirements.txt
   ```

2. **Build CUT3R extension:**

   ```
   cd src/croco/models/curope/
   python setup.py build_ext --inplace
   cd ../../../../ # Return to CUT3R root
   ```

3. **Download checkpoint:**

   ```
   cd src # Navigate to src within CUT3R
   pip install gdown
   gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
   cd ../.. # Return to VLM-3R root
   ```

## ▶️ Test Run

1. **Run Video Test Example:**

   ```
   CUDA_VISIBLE_DEVICES=0 bash scripts/video/demo/video_demo.sh \
       Journey9ni/vlm-3r-llava-qwen2-lora \
       qwen_1_5 32 2 average grid True \
       playground/demo/47334096.mp4 \
       lmms-lab/LLaVA-NeXT-Video-7B-Qwen2
   ```

   **Explanation:**

   - `CUDA_VISIBLE_DEVICES=0`: Specifies the GPU device number to use.
   - `Journey9ni/vlm-3r-llava-qwen2-lora`: Specifies the location of the model checkpoint.
   - `qwen_1_5`: Specifies the model version to use.
   - `32 2 average grid True`: These are parameter settings for model inference.
   - `playground/demo/47334096.mp4`: Specifies the path to the video file to be tested.
   - `lmms-lab/LLaVA-NeXT-Video-7B-Qwen2`: Specifies the base model path for the LoRA model.

2. **Run Image Test Example:**

   ```
   bash scripts/image/demo/image_demo.sh \
       Journey9ni/vlm-3r-llava-qwen2-lora \
       qwen_1_5 2 average grid True \
       playground/demo/scene_47334096_imgs \
       lmms-lab/LLaVA-NeXT-Video-7B-Qwen2
   ```

   **Explanation:**

   - `Journey9ni/vlm-3r-llava-qwen2-lora`: Specifies the location of the model checkpoint.
   - `qwen_1_5`: Specifies the model version to use.
   - `2 average grid True`: These are parameter settings for model inference.
   - `playground/demo/scene_47334096_imgs`: Specifies the path to the directory with image files.
   - `lmms-lab/LLaVA-NeXT-Video-7B-Qwen2`: Specifies the base model path for the LoRA model.

## 📥 Model Weights

The model weights can be downloaded from Hugging Face:

```
# Download model weights from Hugging Face
git lfs install
git clone https://huggingface.co/Journey9ni/vlm-3r-llava-qwen2-lora
```

The model weights include:

- LoRA weight files
- Configuration files
- Other necessary model files

## 🚀 Training

For detailed instructions on training the VLM-3R model, please refer to our primary training script as an example: `scripts/VLM_3R/train_vsibench.sh`.

```bash
# Example training command. Please see the script for more details.
bash scripts/VLM_3R/train_vsibench.sh
```

**Important Note on Video Data:** We do not provide the raw video data from datasets like ScanNet, ScanNet++, or ARKitScenes. You will need to download and process them yourself. The training scripts expect the video data to follow a specific path structure. For instance, the anticipated path for a ScanNet video should be `data/vlm_3r_data/scannet/videos/scene0191_00.mp4`.

**Optional: Pre-extracting Spatial Features**
To significantly accelerate the training process, you can pre-extract spatial features from all your videos beforehand. This avoids redundant feature computation during each training epoch. You can use the provided script for this purpose:

```bash
# Example command for feature extraction
python scripts/extract_spatial_features.py \\
    --input-dir /path/to/your/video/dataset \\
    --output-dir /path/to/save/extracted_features \\
    --cut3r-weights-path /path/to/your/cut3r_weights.pth \\
    --processor-config-path /path/to/your/processor_config.json \\
    --gpu-ids 0,1,2,3
```
Please see the script for a full list of arguments. You will need to create the `processor_config.json` file with the following content:
```json
{
  "do_convert_rgb": null,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "SiglipImageProcessor",
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "processor_class": "LlavaProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 384,
    "width": 384
  }
}
```
After extracting the features, remember to update your training configuration to load these pre-computed features instead of processing raw videos.

Make sure to configure the paths to your video data, benchmark datasets, and desired model output directories within the script.

## 📈 Evaluation

To run the evaluation, first set up the environment:

```bash
cd thinking-in-space # Ensure you are in the correct directory if it's a submodule

conda create --name vsibench python=3.10 -y
conda activate vsibench
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
# Note: The FlashAttention wheel URL might be specific. Consider verifying compatibility.
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install transformers==4.40.0 peft==0.10.0 google-generativeai google-genai huggingface_hub[hf_xet]
```

Then, you can run the evaluation scripts for the VSiBench and VSTiBench benchmarks.

**To evaluate on VSiBench:**
```bash
bash eval_vlm_3r_vsibench.sh
```

**To evaluate on VSTiBench:**
```bash
bash eval_vlm_3r_vstibench.sh
```

## 📝 TODO List

- [x] Release model weights and inference code
- [x] Evaluate on VSiBench
- [x] Release data generation scripts (Note: script for VSiBench's `route plan` task is pending).
- [x] Release training data and training scripts
- [x] Release VSTiBench data and evaluation code

## 🙏 Acknowledgements

We would like to express our gratitude to the following projects for their valuable contributions:

- [CUT3R](https://github.com/CUT3R/CUT3R): Provides the spatial feature encoder used in our model.
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): Serves as the foundation for our codebase.
- [thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space): Offers important evaluation methods for 3D understanding capabilities of VLM.

## 📜 Citation

If you find VLM-3R useful for your research, please consider citing our paper:

```bibtex
@misc{fan2025vlm3rvisionlanguagemodelsaugmented,
      title={VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction}, 
      author={Zhiwen Fan and Jian Zhang and Renjie Li and Junge Zhang and Runjin Chen and Hezhen Hu and Kevin Wang and Huaizhi Qu and Dilin Wang and Zhicheng Yan and Hongyu Xu and Justin Theiss and Tianlong Chen and Jiachen Li and Zhengzhong Tu and Zhangyang Wang and Rakesh Ranjan},
      year={2025},
      eprint={2505.20279},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.20279}, 
}
```
