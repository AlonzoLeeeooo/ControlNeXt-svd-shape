<div align="center">

# Re-implementation of ControlNeXt with Shape Masks on SVD

An unofficial re-implementation of ControlNeXt with shape masks based on the SVD foundation model. Please refer to [this link](https://github.com/dvlab-research/ControlNeXt) for the official implementation of ControlNeXt.

[[`Paper`]](https://arxiv.org/abs/2408.06070) [[`Official Implementation`]](https://github.com/dvlab-research/ControlNeXt) [[`Hugging Face`]](https://huggingface.co/AlonzoLeeeooo/ControlNeXt-svd-shape)
</div>

<!-- omit in toc -->
# Table of Contents
- [<u>1. Overview</u>](#overview)
- [<u>2. To-Do List</u>](#to-do-list)
- [<u>3. Code Structure</u>](#code-structure)
- [<u>4. Implementation Details</u>](#implementation-details)
- [<u>5. Prerequisites</u>](#prerequisites)
- [<u>6. Training</u>](#training)
- [<u>7. Sampling</u>](#sampling)
- [<u>8. Results</u>](#results)
- [<u>9. Star History</u>](#star-history)

<!-- omit in toc -->
# Overview
This is a re-implementation of ControlNet trained with shape masks.
If you have any suggestions about this repo, please feel free to [start a new issue](https://github.com/AlonzoLeeeooo/shape-guided-controlnet/issues/new) or [propose a PR](https://github.com/AlonzoLeeeooo/shape-guided-controlnet/pulls).

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# To-Do List
- [x] Update basic documents
- [x] Update training and inference code
- [x] Update pre-trained model weights
- Regular Maintainence

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Code Structure
```
ControlNeXt-svd-shape
├── LICENSE
├── README.md
├── dataset_loaders                  <----- Code of dataset functions
│   └── youtube_vos.py
├── inference_svd.py                 <----- Script to inference ControlNeXt model
├── models                           <----- Code of U-net and ControlNeXt models
├── pipeline                         <----- Code of pipeline functions
├── requirements.txt                 <----- Dependency list
├── runners                          <----- Code of runner functions
│   ├── __init__.py
│   ├── controlnext_inference_runner.py
│   └── controlnext_train_runner.py
├── train_svd.py                     <----- Script to train ControlNeXt model
└── utils                            <----- Code of toolkit functions
```

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Implementation Details
This re-implementation of ControlNeXt is trained on YouTube-VOS dataset.
The official segmentation annotation of YouTube-VOS is used as the input condition of ControlNeXt.
The overall pipeline is trained with `20,000` iterations, a batch size of `4`, and `bfloat16` precision to achieve its best performance.
For optimization, the trainable parameters contain all `to_k` and `to_v` linear layers in the U-net and the model parameters from ControlNeXt, where this is different from the official implementation that unlocks the entire U-net.

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Prerequisites
1. To install all the dependencies, you can run the one-click installation command line:
```bash
pip install -r requirements.txt
```
2. Download YouTube-VOS from [this link](https://codalab.lisn.upsaclay.fr/competitions/7685#participate-get_data) to prepare the training data.
3. To prepare the pre-trained model weights of Stable Video Diffusion from [this link](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main). For our pre-trained ControlNeXt and U-net weights, you can refer to our [HuggingFace repo](https://huggingface.co/AlonzoLeeeooo/ControlNeXt-svd-shape/tree/main).

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Training
Once the data and pre-trained model weights are ready, you can train the ControlNeXt model with the following command:
```bash
python train_svd.py --pretrained_model_name_or_path SVD_CHECKPOINTS_PATH --train_batch_size TRAIN_BATCH_SIZE --video_path YOUTUBE_VOS_FRAMES_PATH --shape_path YOUTUBE_VOS_ANNOTATION_PATH --output_dir OUTPUT_PATH --finetune_unet
```
You can refer to the following example command line:
```bash
python train_svd.py --pretrained_model_name_or_path checkpoints/svd_xt_1.1 --train_batch_size 4 --video_path youtube_vos/JPEGImages --shape_path youtube_vos/Annotations --annotation_path youtube_vos/meta.json --output_dir OUTPUT_PATH --finetune_unet
```

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)



<!-- omit in toc -->
# Sampling
Once the ControlNeXt model is trained, you can inference it with the following command line:
```bash
python inference_svd.py --pretrained_model_name_or_path SVD_CHECKPOINTS_PATH --validation_control_images_folder INPUT_CONDITIONS_PATH --output_dir OUTPUT_PATH --checkpoint_dir CONTROLNEXT_PATH --ref_image_path REFERENCE_IMAGE_PATH
```
Note that the code differs from the official implementation that you do not need to merge the DeepSpeed checkpoint by running an additional script. All you need is to configure `--checkpoint_dir`.
Normally, a checkpoint saved with the `DeepSpeed` engine should have similar structures as follows:
```
checkpoints
├── latest
├── pytorch_model
│   ├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
│   └── mp_rank_00_model_states.pt
├── random_states_0.pkl
├── scheduler.bin
└── zero_to_fp32.py
```
You need to configure `--checkpoint_dir checkpoints/pytorch_model/mp_rank_00_model_states.pt`, and allow the script to automatically convert the checkpoint to the format of `pytorch_model.bin`.
You can refer to the following example command line:
```bash
python inference_svd.py --pretrained_model_name_or_path checkpoints/svd_xt_1.1 --validation_control_images_folder examples/frames/car --output_dir outputs/inference --checkpoint_dir checkpoints/pytorch_model/mp_rank_00_model_states.pt --ref_image_path examples/frames/car/00000.png
```

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)



<!-- omit in toc -->
# Star History

<p align="center">
    <a href="hhttps://api.star-history.com/svg?repos=alonzoleeeooo/ControlNeXt-svd-shape&type=Date" target="_blank">
        <img width="550" src="https://api.star-history.com/svg?repos=alonzoleeeooo/ControlNeXt-svd-shape&type=Date" alt="Star History Chart">
    </a>
</p>

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)
