# Ensemble

This section of the project deals with running a CNN ensemble on a dataset of images.
![alt text](https://github.com/usef-kh/EC523-Deep-Learning-Project/blob/master/ensemble/ensemble_model.PNG)

## Models

This repo supports a number of different networks. 
- sub1_basic, sub1_tuned
- sub2_basic, sub2_tuned
- sub3_basic, sub3_tuned
- vgg
- Ensemble (sub1, sub2, sub3, vgg)

The actual architectures are all detailed in the <code>models</code> directory

## Data 

We used two datasets for this part of the project
- [Fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [CKPlus](https://www.kaggle.com/shawon10/ckplus)

These datasets must be placed in the root directory in a folder called <code>datasets/fer2013/fer2013.csv</code> and <code>datasets/ckplus/*</code>

Within the <code>data</code> directory, we have a seperate file for loading, prepping, and processing the actual dataset, these are <code>fer2013.py</code> and <code>ckplus.py</code>
They both call onto <code>dataset.py</code> to build a pytorch custom dataset.

Though we did run the project on CKPlus, our work was mostly concerned with Fer2013 

## Environment

Everything was built and run on a conda env 

## Training

To train a network a sample command would be the following
<code>python train.py network=sub1_tuned name=my_sub1</code>

for more training examples and details, checkout <code>scripts/*.sh</code>. These are someof the ones we run to train our networks.

While training, the code will use a number of different files from the <code>utils</code> directory. These are just some supplementary files that are supposed to make it easier for the user to train different variations of a model or load a model from a certain checkpoint.

