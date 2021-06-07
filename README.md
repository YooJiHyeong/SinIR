
# SinIR (Official Implementation)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
We used **Python 3.7.4** and *f-strings* which are introduced in **python 3.6+**

## Training

To train a model, write a proper yaml config file in **'config_train'** folder (sample yaml files provided in the **config_train** folder), and run this command:

```train
python train.py <gpu_num> -y <yaml_file_in_'config_train'_folder>
```
For example, if you want to train a model with **config_train/photo.yaml** on **gpu 0**, run:

```train_ex
python train.py 0 -y photo
```
This will output **a trained model, training logs, training output images** and so on, to a subdirectory of **'outs'** folder with proper naming and numbering which are used for inference.

Note that even though we provide one yaml file for each task, they can be used interchangeably, except few tasks.

You can copy and modify them depending on your purpose. Detailed explanation about configuration is written in the sample yaml files. Please read through it carefully if you need.


## Inference

To carry out inference (*i.e.*, image manipulation), you can specify inference yaml files in training yaml files.
Please see provided sample training yaml files.

Or alternatively you can run this command:

```infer
python infer.py <output_dirnum> <gpu_num> -y <yaml_file_in_config_folder>
```
For example, if you want to carry out inference with a trained model numbered **002**, with **config_infer/photo_infer.yaml** on **gpu 0**, run:

```infer_ex
python infer.py 2 0 -y photo_infer
```
Then it will automatically find an output folder numbered **002** and conduct image manipulation, saving related results in the subdirectory.



Note that duplicated numbering (which can be avoided with a normal usage) will incur error. In this case, please keep only one output folder.

We also provide sample yaml files for inference which are paired with yaml files for training. Feel free to copy and modify depending on your purpose.


## Acknowledgement
This repository includes images from:
<ol>
	<li> https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/ (BSD dataset)
	<li> https://github.com/luanfujun/deep-painterly-harmonization/ (https://arxiv.org/abs/1804.03189)
	<li> https://github.com/luanfujun/deep-photo-styletransfer (https://arxiv.org/abs/1703.07511)
	<li> The Web (free images)
</ol>

This repository includes codes snippets from:
<ol>
	<li> <strong>SSIM</strong>: https://github.com/VainF/pytorch-msssim
	<li> <strong>Anti-aliasing + Bicubic resampling</strong>: https://github.com/thstkdgus35/bicubic_pytorch
	<li> <strong>dilated mask</strong>: https://github.com/tamarott/SinGAN
</ol>
