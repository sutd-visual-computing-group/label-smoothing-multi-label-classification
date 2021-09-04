import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading the dataset
df = pd.read_csv('scores_wth_gt-1_mobnet_0.csv') #change into the correct csv file name
df2 = pd.read_csv('scores_wth_gt-1_mobnet_01.csv')
df3 = pd.read_csv('scores_wth_gt-1_mobnet_02.csv')

# checking features
groundtruth = df['gt'].tolist()
# display variables
#print(groundtruth)

ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

dataframes = list(zip(classes,ids))
#print(dataframes)

disallowed_characters = ']" [' #characters to strip off

encodingpositions=[]

#Multi hot encoding
listindex2=[]

for elem in groundtruth:
    listindex=[]
    for character in disallowed_characters:
	    elem = elem.replace(character, "")
    elem = elem.split(",")
    for minielem in elem:
        for a in dataframes:
            if minielem == "'"+a[0]+"'":
                listindex.append(a[1])
                listindex2.append(a[1])
    encodingpositions.append(listindex)

listindex2 = np.array(listindex2)

onehot_encoded=[]

for element in encodingpositions:
    zeroed = [0 for _ in range(0,20,1)]
    for b in element:
        zeroed[b] = 1
    onehot_encoded.append(zeroed)

onehot_encoded = np.array(onehot_encoded)
print(onehot_encoded)
print(onehot_encoded.shape)
print(type(onehot_encoded))

############################################################
from sklearn.preprocessing import label_binarize

# Use label_binarize to use multi-label like settings
Y = label_binarize(listindex2, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
n_classes = Y.shape[1]

#print(Y)
#print(type(Y))
#print(Y.shape)
###########################################################

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

cat = df.iloc[:, 2:22]
#print(cat)

df_list = cat.values.tolist()
df_list = np.array(df_list)
print(df_list.shape)
#print(df_list)
#print(type(df_list))
df_list2 = df2.iloc[:, 2:22].values.tolist()
df_list2 = np.array(df_list2)

df_list3 = df3.iloc[:, 2:22].values.tolist()
df_list3 = np.array(df_list3)

#GT is the Ground truth label, y_pred is the predictions label
#GT = onehot_encoded, y_pred = df_list
#Format: precision_recall_curve(GT[:,i],y_pred[:,i])

# Label Smoothing 0.0
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(onehot_encoded[:, i],
                                                        df_list[:, i])
    average_precision[i] = average_precision_score(onehot_encoded[:, i], df_list[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(onehot_encoded.ravel(),
    df_list.ravel())
average_precision["micro"] = average_precision_score(onehot_encoded, df_list,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes for ls 0.0: {0:0.2f}'
      .format(average_precision["micro"]))
#######################################################
# Label Smoothing 0.01
precision2 = dict()
recall2 = dict()
average_precision2 = dict()
for i in range(n_classes):
    precision2[i], recall2[i], _ = precision_recall_curve(onehot_encoded[:, i],
                                                        df_list2[:, i])
    average_precision2[i] = average_precision_score(onehot_encoded[:, i], df_list2[:, i])

# A "micro-average": quantifying score on all classes jointly
precision2["micro"], recall2["micro"], _ = precision_recall_curve(onehot_encoded.ravel(),
    df_list2.ravel())
average_precision2["micro"] = average_precision_score(onehot_encoded, df_list2,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes for ls 0.1: {0:0.2f}'
      .format(average_precision2["micro"]))
#####################################################################
# Label Smoothing 0.02
precision3 = dict()
recall3 = dict()
average_precision3 = dict()
for i in range(n_classes):
    precision3[i], recall3[i], _ = precision_recall_curve(onehot_encoded[:, i],
                                                        df_list3[:, i])
    average_precision3[i] = average_precision_score(onehot_encoded[:, i], df_list3[:, i])

# A "micro-average": quantifying score on all classes jointly
precision3["micro"], recall3["micro"], _ = precision_recall_curve(onehot_encoded.ravel(),
    df_list3.ravel())
average_precision3["micro"] = average_precision_score(onehot_encoded, df_list3,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes for ls 0.2: {0:0.2f}'
      .format(average_precision3["micro"]))

#multi-figure approach
plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')
plt.step(recall2['micro'], precision2['micro'], where='post')
plt.step(recall3['micro'], precision3['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.1])
plt.title('Precision-Recall Curve (MobileNetV2)', fontsize=20)
#print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
plt.legend(['\u03B1 = 0.0','\u03B1 = 0.1','\u03B1 = 0.2'], fontsize=18)
plt.show()

""" Optional Code from sckitlearn website to plot all 20 classes AP on a single graph
from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.show()
"""
print("\nAverage Precision for Each Class with Label Smoothing 0.0:\n")
print(average_precision)
print("\nAverage Precision for Each Class with Label Smoothing 0.1:\n")
print(average_precision2)
print("\nAverage Precision for Each Class with Label Smoothing 0.2:\n")
print(average_precision3)