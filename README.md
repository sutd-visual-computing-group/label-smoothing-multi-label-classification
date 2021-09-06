# Introduction
This GitHub repository contains Python code on **Label Smoothing**, it is built based on a repository: https://github.com/keshik6/pascal-voc-classification. The idea here is to apply some modifications on the encode_labels function under utils.py to take into account multi-hot vector encoding and integrating with a label smoothed system as well as try out some experiment on constructing precision-recall curves. The main objectives involves:
* Implement two different **Label Smoothing Schemes** (with and without added confidence towards the probabilities of incorrect classes) and analyze the Average Precision results for different ResNet models (ResNet-18,34,50) on multi-label classification for the Pascal VOC dataset
* Perform quick analysis and construct a **Precision and Recall curve** from given .csv files for different degrees of Label Smoothing (see prcurve.py)

## Dataset
The dataset used in the experiment is **Pascal VOC 2012** dataset which is built-in on the latest version of pytorch, separated into the Training, Validation, and Test sets. The Pascal VOC 2012 dataset contains 20 object classes divided into 4 main groups:
1. Person
2. Bird, cat, cow, dog, horse, sheep
3. Aeroplane, bicycle, boat, bus, car, motorbike, train
4. Bottle, chair, dining table, potted plant, sofa, tv/ monitor

## Loss function

The task is multi-label classification for 20 object classes, which is analogous to creating 20 object detectors, 1 for every class. The loss function used is the binary cross entropy (with logits loss), where in PyTorch, the loss function can be applied using torch.nn.BCEWithLogitsLoss( ). Do note that this function provides numerical stability over the sequence of sigmoid followed by binary cross entropy. The loss function is clearly documented at ***https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCEWithLogitsLoss***

## Metrics
**Average precision** is used as the metric to measure performance which is the average of the maximum precisions at different recall values. 

## Model
The models used here are **Residual Neural Networks** with varying depths of 18, 34, and 50 layers, trained and tested on a local machine. The models area trained with a batch size of 16, learning rate of 1.5e-4 for the ResNet backbone and 5e-2 for ResNet Fully-Connected layers. The degree of label smoothing used here is 0.1 for all experiments.

## Label Smoothing
Label smoothing is a regularization technique which turns hard class labels assignments into soft label assignments, it operates directly on the label themselves and may lead to a better generalization [1]. Labels in the scope of LS are usually classified into two types:
* **Hard label assignments**, all entries in the matrix/vector are 0 except the one corresponding to the correct class or classes in the case of Multi-Hot encoding which is assigned to 1
* **Soft label assignments**, the correct class or classes have the largest probability and all other classes have a very small probability but not zero, there is assignment of probabilities to the incorrect classes.

There are two different Label Smoothing schemes utilized in this experiment:
1. **Label Smoothing Scheme 1:** The probabilities of correct classes are decreased by a certain degree of LS, the probabilities of other classes are increased by a small value of the degree divided by the total number of object classes
2. **Label Smoothing Scheme 2:** The probabilities of correct classes are decreased by the LS degree, the probabilities of incorrect classes stays at zero

One reason of using LS is to prevent the model from becoming too confident in its predictions and reduce overfitting

## Results
### Label Smoothing Scheme 1
**Average Precision**
| Model     | Average Precision (Training Set)|Average Precision (Validation Set)|Average Precision (Test Set)|
| --------- | --------- | --------- | --------- |
| ResNet-18 | 0.916 | 0.853 | 0.865 |
| ResNet-34 | 0.977 | 0.864 | 0.874 |
| ResNet-50 | 0.992 | 0.871 | 0.883 |

**Losses**
| Model	   | Loss (Training Set) | Loss (Validation Set) | Loss (Test Set)|
| --------- | --------- | --------- | --------- |
| ResNet-18	| 1.5020	| 2.3912	| 2.1982 |
| ResNet-34	| 0.7101	| 2.6393	| 2.3846 |
| ResNet-50	| 0.4265	| 2.7362	| 2.4859 |

The results shows that with the same degree of label smoothing, a ResNet-18 model with label smoothing degree 0.1 obtains an average test precision of 0.865 and test loss of 2.1982, where the average precision is lower than that of the ResNet-34 and ResNet-50, on the validation dataset the average precision is also lower than the deeper models indicating that the model may have less complexity and thus unable to capture the features of the image as well as the deeper models but it is less likely to overfit.

The results for ResNet-50 obtains an average test precision of 0.883 which is higher than ResNet-18 as well as ResNet-34 for Label Smoothing Scheme 1 with degree 0.1, and the test loss is 2.4859 which is higher than ResNet-18 and 34. Deeper models may be able to capture more features with increased complexity however do take note that models that overfit may perform poorly. For Deep Neural Networks, as depth increases the average precision for training also increases from ResNet-18 to ResNet-50 and the same goes for the average precision of the validation set. 

### Label Smoothing Scheme 2
**Average Precision**
| Model     | Average Precision (Training Set)|Average Precision (Validation Set)|Average Precision (Test Set)|
| --------- | --------- | --------- | --------- |
| ResNet-18	| 0.979 | 0.865 | 0.873 |
| ResNet-34	| 0.982 | 0.847 | 0.859 |
| ResNet-50	| 0.970 | 0.863 | 0.872 |

**Losses**
| Model	   | Loss (Training Set) | Loss (Validation Set) | Loss (Test Set)|
| --------- | --------- | --------- | --------- |
| ResNet-18	| 0.7318	| 2.4773	| 2.2921 |
| ResNet-34	| 0.6018	| 2.9692	| 2.6780 |
| ResNet-50	| 0.8527	| 2.5462	| 2.3226 |

By modifying and changing the label smoothing function to only reduce the probabilities of the correct classes while keeping incorrect classes at 0, the average test precision obtained for ResNet-18 is 0.873 which is higher compared to the same model with label smoothing scheme 1 (the average test precision is 0.865), the validation and training average precisions are 0.979 and 0.865 respectively which is also higher than scheme 1, thus for the ResNet-18 model, keeping the incorrect classes at 0 and softening the correct classes by a small degree may help increase the model performance since the model itself is not too deep compared to the two other variants and is less prone to overfitting.

Using label smoothing scheme 2, the results for ResNet-34 performs worse than LS scheme 1 where the average precision for test set is 0.859 as opposed to 0.874, the validation average precision is also lower with 0.847 as opposed to 0.864, only the training average precision is higher being 0.982. A similar trend can be seen on the results for ResNet-50 which also performs worse than LS scheme 1, the average precision for test set is 0.872 as opposed to 0.883, the validation average precision is also lower with 0.863 as opposed to 0.871, furthermore the training average precision is also lower with a value of 0.970 instead of 0.992. Here we are totally ignoring the probabilities of the incorrect classes, thus models are more prone to overfitting and too confident in the correct classes.

## Insights
* The degree of Label Smoothing chosen to be 0.1 since it is the most widely used degree which maintains a decently high confidence on the correct classes and adds a bit to the incorrect classes
* The performance of the models after label smoothing is reliant on the training dataset which is Pascal VOC 2012 that is widely used in detection tasks
* Residual neural networks are chosen as they are able to utilize deep and large number of layers but remain less complex than most other models, signifying compactness

## Potential Problems
* The model is only trained on the Pascal VOC 2012 Dataset and while it is a vast dataset, there are object classes not included within the dataset which will affect the classification performance on these foreign object classes
* A single degree of label smoothing chosen to be 0.1 is used on all of the experiments which do not truly shows the changes if this particular degree is modified
* The changes in the average precision between ResNet-18, ResNet-34, and ResNet-50 is not that significant thus using a deeper model or architecture which has higher risk of overfitting may not always be ideal

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
