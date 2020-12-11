# AudioVisual

This section of the project deals with running a CNN ensemble on a dataset of images.

## Models

This repo supports a number of different networks. 
- cnn2d
- cnn3d
- elm1
- elm2
- svm

The actual architectures are all detailed in the <code>models</code> directory
SVM, on the other hand is an sklearn model that is defined in <code>train_svm.py</code> 

## Data 

The dataset used for this part of the project is [eNTERFACE'05](http://www.enterface.net/enterface05/)

This dataset must be placed in the root directory in a folder called <code>datasets/enterface/original/</code>
To extract the audio files, you must run <code>convert_avi_wav.py</code>

Due to the time it takes to preprocess this dataset, we created a file <code>process_data.py</code> that will perform all key frame selections and compute the requires spectrograms along with all other preprocessing required

This file will call upon all relevant files within  the <code>data</code> directory to do this. Primarity, <code>enterface.py</code> will load and prep all files, <code>processor.py</code> will perform the preprocessing, and <code>dataset.py</code> will build a pytorch custom dataset.

The processed data must be placed in <code>datasets/enterface/processed</code>

## Environment

Everything was built and run on a conda env 

## Training

Unlike in the CNN Ensemble, depending on the network chosen, the training inputs and targets are different in this section of the project.
For this, we have several training files. Depending on the network chosen, one of the following will need to be run.
It is not possible to train without performing the required data steps above to preprocess the data.

### CNNs
- <code>python train_cnn.py network=cnn2d name=my_cnn2d</code>
- <code>python train_cnn.py network=cnn3d name=my_cnn3d</code>

### ELMs
- <code>python train_elm.py network=elm1 name=my_elm1 cnn2d_path=PATH_TO_CNN2D_CHECKPOINT cnn3d_path=PATH_TO_CNN3D_CHECKPOINT</code>
- <code>python train_elm.py network=elm2 name=my_elm2 elm1_path=PATH_TO_ELM1_CHECKPOINT</code>

### SVM
- <code>python train_svm.py network=svm name=my_svm elm2_path=PATH_TO_ELM2_CHECKPOINT</code>

