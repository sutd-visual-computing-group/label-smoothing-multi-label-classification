# Introduction
This GitHub repository contains Python code on Label Smoothing, it is built based on a repository created by my mentor during my Internship: https://github.com/keshik6/pascal-voc-classification. The idea here is to apply some modifications on the encode_labels function under utils.py to take into account multi-hot vector encoding and integrating with a label smoothed system as well as try out some experiment on constructing precision-recall curves. The main objectives involves:
* Implement two different Label Smoothing Schemes (with and without added confidence towards the probabilities of incorrect classes) and analyze the Average Precision results for different ResNet models (ResNet-18,34,50) on multi-label classification for the Pascal VOC dataset
* Perform quick analysis and construct a Precision and Recall curve from given .csv files for different degrees of Label Smoothing

## Dataset
The dataset used in the experiment is Pascal VOC 2012 dataset which is built-in on the latest version of pytorch, separated into the Training, Validation, and Test sets. The Pascal VOC 2012 dataset contains 20 object classes divided into 4 main groups:
1. Person
2. Bird, cat, cow, dog, horse, sheep
3. Aeroplane, bicycle, boat, bus, car, motorbike, train
4. Bottle, chair, dining table, potted plant, sofa, tv/ monitor

## Loss function

The task is multi-label classification for 20 object classes, which is analogous to creating 20 object detectors, 1 for every class. The loss function used is the binary cross entropy (with logits loss), where in PyTorch, the loss function can be applied using torch.nn.BCEWithLogitsLoss( ). Do note that this function provides numerical stability over the sequence of sigmoid followed by binary cross entropy. The loss function is clearly documented at ***https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCEWithLogitsLoss***

## Metrics
Average precision is used as the metric to measure performance which is the average of the maximum precisions at different recall values. 

## Model
The models used here are Residual Neural Networks with varying depths of 18, 34, and 50 layers, trained and tested on a local machine. The models area trained with a batch size of 16, learning rate of 1.5e-4 for the ResNet backbone and 5e-2 for ResNet-fc.

## Challenges


## Results
### Training History
![loss-1](https://user-images.githubusercontent.com/21276946/55679999-d3f2d380-5946-11e9-92a4-45b3f0356ab9.png)
![accuracy-1](https://user-images.githubusercontent.com/21276946/55680003-d7865a80-5946-11e9-80a4-14898c4c88e7.png)

## How to reproduce the code?
### Pytorch source code
1. Install dependencies: pip install -r requirements.txt
2. Directory structure
    * /docs: contain project and devkit documentation
    * /models: contains model weights, log-files and plots
    * /src: contains source code
    * /data: data directory to download and extract pascal VOC dataset (You should create this directory manually)
    * Run the main function in main.py with required arguments. The codebase is clearly documented with clear details on how to execute the functions. It also includes an example. You need to interface only with this function to reproduce the code.

## Acknowledgements
This work was done by Leon Tjandra and Keshigeyan Chandrasegaran at Temasek Laboratories, Singapore University of Technology and Design.
