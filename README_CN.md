GFPGAN (CVPR 2021)
download PyPI Open issue Closed issue LICENSE python lint Publish-pip

Colab Demo for GFPGAN google colab logo; (Another Colab Demo for the original paper model)
Online demo: Huggingface (return only the cropped face)
Online demo: Replicate.ai (may need to sign in, return the whole image)
We provide a clean version of GFPGAN, which can run without CUDA extensions. So that it can run in Windows or on CPU mode.
🚀 Thanks for your interest in our work. You may also want to check our new updates on the tiny models for anime images and videos in Real-ESRGAN 😊

GFPGAN aims at developing a Practical Algorithm for Real-world Face Restoration.
It leverages rich and diverse priors encapsulated in a pretrained face GAN (e.g., StyleGAN2) for blind face restoration.

🚩 Updates

✅ Integrated to Huggingface Spaces with Gradio. See Gradio Web Demo.
✅ Support enhancing non-face regions (background) with Real-ESRGAN.
✅ We provide a clean version of GFPGAN, which does not require CUDA extensions.
✅ We provide an updated model without colorizing faces.
If GFPGAN is helpful in your photos/projects, please help to ⭐ this repo or recommend it to your friends. Thanks😊 Other recommended projects:
▶️ Real-ESRGAN: A practical algorithm for general image restoration
▶️ BasicSR: An open-source image and video restoration toolbox
▶️ facexlib: A collection that provides useful face-relation functions
▶️ HandyView: A PyQt5-based image viewer that is handy for view and comparison

📖 GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior
[Paper]   [Project Page]   [Demo]
Xintao Wang, Yu Li, Honglun Zhang, Ying Shan
Applied Research Center (ARC), Tencent PCG



🔧 Dependencies and Installation
Python >= 3.7 (Recommend to use Anaconda or Miniconda)
PyTorch >= 1.7
Option: NVIDIA GPU + CUDA
Option: Linux
Installation
We now provide a clean version of GFPGAN, which does not require customized CUDA extensions.
If you want to use the original model in our paper, please see PaperModel.md for installation.

Clone repo

git clone https://github.com/TencentARC/GFPGAN.git
cd GFPGAN
Install dependent packages

# Install basicsr - https://github.com/xinntao/BasicSR
# We use BasicSR for both training and inference
pip install basicsr

# Install facexlib - https://github.com/xinntao/facexlib
# We use face detection and face restoration helper in the facexlib package
pip install facexlib

pip install -r requirements.txt
python setup.py develop

# If you want to enhance the background (non-face) regions with Real-ESRGAN,
# you also need to install the realesrgan package
pip install realesrgan
⚡ Quick Inference
Download pre-trained models: GFPGANCleanv1-NoCE-C2.pth

wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth -P experiments/pretrained_models
Inference!

python inference_gfpgan.py --upscale 2 --test_path inputs/whole_imgs --save_root results
If you want to use the original model in our paper, please see PaperModel.md for installation and inference.

🏰 Model Zoo
GFPGANCleanv1-NoCE-C2.pth: No colorization; no CUDA extensions are required. It is still in training. Trained with more data with pre-processing.
GFPGANv1.pth: The paper model, with colorization.
You can find more models (such as the discriminators) here: [Google Drive], OR [Tencent Cloud 腾讯微云]

💻 Training
We provide the training codes for GFPGAN (used in our paper).
You could improve it according to your own needs.

Tips

More high quality faces can improve the restoration quality.
You may need to perform some pre-processing, such as beauty makeup.
Procedures

(You can try a simple version ( options/train_gfpgan_v1_simple.yml) that does not require face component landmarks.)

Dataset preparation: FFHQ

Download pre-trained models and other data. Put them in the experiments/pretrained_models folder.

Pre-trained StyleGAN2 model: StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth
Component locations of FFHQ: FFHQ_eye_mouth_landmarks_512.pth
A simple ArcFace model: arcface_resnet18.pth
Modify the configuration file options/train_gfpgan_v1.yml accordingly.

Training

python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch

📜 License and Acknowledgement
GFPGAN is released under Apache License Version 2.0.

BibTeX
@InProceedings{wang2021gfpgan,
    author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
    title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
📧 Contact
If you have any question, please email xintao.wang@outlook.com or xintaowang@tencent.com.
