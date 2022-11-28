# Eigen-Face

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
