# Introduction
This GitHub repository contains Python code on **Label Smoothing**, it is built based on a repository: https://github.com/keshik6/pascal-voc-classification. The idea here is to apply some modifications on the encode_labels function under utils.py to take into account multi-hot vector encoding and integrating with a label smoothed system as well as try out some experiment on constructing precision-recall curves. The main objectives involves:
* Implement two different **Label Smoothing Schemes** (with and without added confidence towards the probabilities of incorrect classes) and analyze the Average Precision results for different ResNet models (predominantly ResNet-18) as well as the MobileNet-V2 model on multi-label classification for the Pascal VOC dataset
* Perform quick analysis and construct a **Precision and Recall curve** from given .csv files for different degrees of Label Smoothing (see prcurve.py)

## Dataset
The dataset used in the experiment is **Pascal VOC 2012** dataset which is built-in on the latest version of pytorch, separated into the Training and Validation sets. The Pascal VOC 2012 dataset contains 20 object classes divided into 4 main groups:
1. Person
2. Bird, cat, cow, dog, horse, sheep
3. Aeroplane, bicycle, boat, bus, car, motorbike, train
4. Bottle, chair, dining table, potted plant, sofa, tv/ monitor

This experiment also involves a Test Dataset which is identical to the Validation set, however here we utilize a different transformation towards the dataset which uses a FiveCrop from the torchvision library [2] which crops the given image into four corners and the central crop with a parameter size of 300 denoting the desired output size of the crop.

## Loss function

The task is multi-label classification for 20 object classes, which is analogous to creating 20 object detectors, 1 for every class. The loss function used is the binary cross entropy (with logits loss), where in PyTorch, the loss function can be applied using torch.nn.BCEWithLogitsLoss( ). Do note that this function provides numerical stability over the sequence of sigmoid followed by binary cross entropy. The loss function is clearly documented at ***https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCEWithLogitsLoss***

## Metrics
**Average precision** is used as the metric to measure performance which is the average of the maximum precisions at different recall values. 

## Model
The models used here are **Residual Neural Networks** with varying depths of 18, 34, and 50 layers, trained and tested on a local machine. The models area trained with a batch size of 8, learning rate of 1.5e-4 for the ResNet backbone and 5e-2 for ResNet Fully-Connected layers. The degree of label smoothing used here is 0.1 for all experiments. 

Another model was also utilized here which is the **MobileNet-V2** model which is a convolutional neural network designed for compact mobile devices. The model consists of two different types of blocks: residual block with stride of 1 and a block with a stride of 2 for downsamlpling. The architecture starts off with a convolutional 2d layer followed by series of bottleneck layers, ending with convolutional 2d and an average pooling layer [3]. The model uses the same hyperparameters as the ResNet models with an identical degree of label smoothing as well.

## Label Smoothing
Label smoothing is a regularization technique which turns hard class labels assignments into soft label assignments, it operates directly on the label themselves and may lead to a better generalization [1]. Labels in the scope of LS are usually classified into two types:
* **Hard label assignments**, all entries in the matrix/vector are 0 except the one corresponding to the correct class or classes in the case of Multi-Hot encoding which is assigned to 1
* **Soft label assignments**, the correct class or classes have the largest probability and all other classes have a very small probability but not zero, there is assignment of probabilities to the incorrect classes.

There are two different Label Smoothing schemes utilized in this experiment:
1. **Label Smoothing Scheme 1:** The probabilities of correct classes are decreased by a certain degree of LS, the probabilities of other classes are increased by a small value of the degree divided by the total number of object classes. We can denote the Probability of the correct classes as **P(C)**, the Probabilities of the incorrect classes as **P(I)**, degree of label smoothing as **α**, and the total number of object classes as **n**, then the following equation applies:

<p align="center">
P(C) = 1 - α + (α/n)
</p>
<p align="center">
P(I) = 0 + (α/n)
</p>

2. **Label Smoothing Scheme 2:** The probabilities of correct classes are decreased by the LS degree, the probabilities of incorrect classes stays at zero

<p align="center">
P(C) = 1 - α
</p>
<p align="center">
P(I) = 0
</p>

One reason of using LS is to prevent the model from becoming too confident in its predictions and reduce overfitting

## Results
The main results presented here will be a comparison between the ResNet-18 model as a lighweight model of the Residual Neural Network type alongside the MobileNet-V2 on the different Label Smoothing options (Without Label Smoothing, Scheme 1, and Scheme 2).

### Average Precision (Training Set)
| Model     | Baseline | LS Scheme 1 | LS Scheme 2 |
| --------- | --------- | --------- | --------- |
| ResNet-18 | 0.997 | 0.916 | 0.979 |

### Average Precision (Validation Set)
| Model     | Baseline | LS Scheme 1 | LS Scheme 2 |
| --------- | --------- | --------- | --------- |
| ResNet-18 | 0.872 | 0.853 | 0.865 |

### Average Precision (Test Set)
| Model     | Baseline | LS Scheme 1 | LS Scheme 2 |
| --------- | --------- | --------- | --------- |
| ResNet-18 | 0.882 | 0.865 | 0.873 |

### Loss (ResNet-18)

## Insights
* The degree of Label Smoothing chosen to be 0.1 since it is the most widely used degree which maintains a decently high confidence on the correct classes and adds a bit to the incorrect classes
* The performance of the models after label smoothing is reliant on the training dataset which is Pascal VOC 2012 that is widely used in detection tasks
* Residual neural networks are chosen as they are able to utilize deep and large number of layers but remain less complex than most other models, signifying compactness

## Potential Problems
* The model is only trained on the Pascal VOC 2012 Dataset and while it is a vast dataset, there are object classes not included within the dataset which will affect the classification performance on these foreign object classes
* A single degree of label smoothing chosen to be 0.1 is used on all of the experiments which do not truly shows the changes if this particular degree is modified

## Steps to utilize the code
1. Install dependencies via: pip install -r requirements.txt
2. Directory structure
    * /docs: contain project and devkit documentation (credits to Keshigeyan Chandrasegaran)
    * /models: contains model weights, log-files and plots
    * /src: contains source code
    * /data: data directory to download and extract pascal VOC dataset (Create this directory manually)
    * Run the main function in main.py with required arguments. The codebase is clearly documented with clear details on how to execute the functions. You need to interface only with this function to reproduce the code.

## Acknowledgements
This work was done by Leon Tjandra during an internship at Temasek Laboratories, Singapore University of Technology and Design.

## References
[1] Rosebrock, Adrian. (2019). "Label smoothing with Keras, TensorFlow, and Deep Learning". https://www.pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/

[2] Torch Contributors. (2017). "Torchvision.transforms". https://pytorch.org/vision/stable/transforms.html

[3] Tsang, Sik-Ho. (2019). "Review: MobileNetV2 — Light Weight Model (Image Classification)". https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c
