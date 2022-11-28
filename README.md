# Eigen-Face
## Setup
Download the following datasets and move into `data` directory.
1. [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) for PASCAL VOC dataset for non-face categories pretraining 
2. [AT&T face images](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)
3. [cropped and aligned celeba faces](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=share_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) and [face identity](https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=share_link)
4. our [non-face test images](https://drive.google.com/file/d/16VsYUZaV-36s0RIFuDsL33cMiIPMbwBG/view?usp=share_link)

Also, download our [trained checkpoints](https://drive.google.com/drive/folders/1x00rOhIlZ6WNNc474OwmGKb-IcmMtASW?usp=share_link) for the CNN models for face classification and identification and store inside `checkpoints` directory.

## Dimensionality Reduction and Face Reconstruction
Run the following command for analysis on Eigen-Face, importance of principal components and face reconstruction. 
```shell
python -m src.eigen_face.eigen_face
```

## Face non-face Classification

We build classifiers (LDA, KNN, SVC and RandomForest) to differentiate face images from non-face images by training on a sample of images from [AT&T dataset](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/) and non-face images created by sampling from [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/). PCA is applied to reduce the dimensionality of image representations. To run training and then inference on these datasets, you can run the following command.
```shell
python -m src.face_non_face.test_other_classifiers
```
Also, we train a resnet 18 model for this task. The CNN model is first trained for face non-face classification using face images from [CELEBA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and non-face images from [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/). Then, this model is finetuned in the above mentioned small dataset. To run this pretraining and finetuning pipeline, run the following command. 
```shell
python -m src.face_non_face.train_resnet
```
And, run the following to test finetuned model.
```shell
python -m src.face_non_face.test_resnet
```

## Face Identification
We approach face identification using multi-class classification and embedding space based models. 

First, we train LDA, KNN, SVC and random forest classifiers on [AT&T dataset](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/) on multi-class classification setting. PCA is applied on the images to reduce the dimensionality. You can run the following command to train and evaluate the pipeline. 
```shell
python -m src.face_identification.test_other_identifiers
```
Next, we train resnet 18 using contrastive learning to have faces of same subject to have close embeddings and different subjects to have distant embeddings in cosine space. For this, we pretrain our model on [CELEBA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and finetune on  [AT&T dataset](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/). You can run the following command for the overall pretraining and finetuning process.   
```shell
python -m src.face_identification.train_resnet
```
And, run the following command to evaluate the finetuned model.
```shell
python -m src.face_identification.test_resnet
```
