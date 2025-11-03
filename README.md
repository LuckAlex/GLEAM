# GLEAM

GLEAM: Enhanced Transferable Adversarial Attacks for Vision-Language Pre-training Models via Global-Local Transformations

## Brief Introduction

Vision-language pre-training (VLP) models leverage large-scale cross-modal pre-training to align vision and text modalities, achieving impressive performance on tasks like image-text retrieval and visual grounding. However, these models are highly vulnerable to adversarial attacks, raising critical concerns about their robustness and reliability in safety-critical applications. Existing black-box attack methods are limited by insufficient data augmentation mechanisms or the disruption of global semantic structures, leading to poor adversarial transferability. To address these challenges, we propose the Global-Local Enhanced Adversarial Multimodal attack (GLEAM), a unified framework for generating transferable adversarial examples in vision-language tasks. GLEAM introduces a local feature enhancement module that achieves diverse local deformations while maintaining global semantic and geometric integrity. It also incorporates a global distribution expansion module, which expands feature space coverage through dynamic transformations. Additionally, a cross-modal feature alignment module leverages intermediate adversarial states to guide text perturbations. This enhances cross-modal consistency and adversarial text transferability. Extensive experiments on Flickr30K and MSCOCO datasets show that GLEAM outperforms state-of-the-art methods, with over 10\%-30\% higher attack success rates in image-text retrieval tasks and over 30\% improved transferability on large models like Claude 3.5 Sonnet and GPT-4o. GLEAM provides a robust tool for exposing vulnerabilities in VLP models and offers valuable insights into designing more secure and reliable vision-language systems.<p align="left">

</p>

## Start 

### 1. Install dependencies
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Install required attack libraries
```bash
# Download OpenAttack library 
https://github.com/thunlp/OpenAttack
# Download TransferAttack library
https://github.com/Trustworthy-AI-Group/TransferAttack
```

### 3. Prepare datasets and models

Download the datasets, [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#home) (the annotations is provided in ./data_annotation/). Set the root path of the dataset in `./configs/Retrieval_flickr.yaml, image_root`.  
The checkpoints of the fine-tuned VLP models is accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL), [CLIP](https://huggingface.co/openai/clip-vit-base-patch16).




## Citation

Kindly include a reference to this paper in your publications if it helps your research:
```
@InProceedings{Liu_2025_ICCV,
    author    = {Liu, Yunqi and Ouyang, Xue and Cui, Xiaohui},
    title     = {GLEAM: Enhanced Transferable Adversarial Attacks for Vision-Language Pre-training Models via Global-Local Transformations},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {1665-1674}
}
```
