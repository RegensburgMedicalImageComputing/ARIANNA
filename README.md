# Automated Retinal Image Analysis using a Neural Net Architecture (ARIANNA)                   
Dataset: AREDS (phs00001)    

Phenotype: AMDSEV (Ungradable + 9 Scale + 3 late AMD): AREDS severity of one fundus image

Library: Mxnet 0.9x using CUDA 8.0.44 and cuDNN 8.0; NVIDIA-Linux-x86_64-378.13.run
 
.rec files created by im2rec

Author: Felix Grassmann and Judith Mengelkamp, University of Regensburg/OTH Regensburg 

Version: 1.2 

## 1. Normalize the input images that will be predicted
Use python and normalize all images according to the method proposed by Ben Graham, the winner of the DR prediction Kaggle competion

Go to the directory that contains your images and start python

```python
python

import cv2, glob, numpy
import matplotlib.pyplot as plt
from multiprocessing import Pool
```


This normalization will result in background substracted images with normalized color balance and illumination


Adjust outdir accordingly!

```python
for f in (glob.glob("*.jpg")):
        try:
            a=cv2.imread(f)
            aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),1000/30),-4,128)
            outdir="/path/to/hold/norm/images/"
            cv2.imwrite(outdir+f,aa)
        except:
            print f
```

Use the im2rec program to create rec files for fast i/o

sample_information.lst (no header, just shown for clarity):

> random_ID	class_label	image_name

> 292	7	1199_00_LE_F2_LS.jpg

> 294	11	1199_04_LE_F2_LS.jpg


Change folder to the folder that should have the rec files

```unix
cd /path/to/rec_files/
im2rec sample_information.lst /path/to/hold/norm/images/  pred.rec resize=512
```

## 2. Predict the class of each image using CNNs implemented in MXNet

```python
python
```

Import the important libraries

```python
from __future__ import absolute_import
from __future__ import print_function
from collections import OrderedDict, namedtuple
import numpy as np
import re
import mxnet as mx
```


Set the devices that you will use for prediction (GPU or CPU)

```python
ctx=[mx.gpu(0),mx.gpu(1),mx.gpu(2)]
ctx=[mx.gpu(1)]
```

Setup a data iterator that will provide the data to the neural networks

```python
dataiter_pred = mx.io.ImageRecordIter(
	path_imgrec="/path/to/rec_files/pred.rec",
	data_shape=(3,512,512),
	path_imglist="/path/to/hold/rec_files/sample_information.lst",
	batch_size=1,
	preprocess_threads=10,
)
```

Load the respective model from file (requires the checkpoint and the .json file in the same directory)
predict the classes of the images and save to text

Note: the names of the text files should be as stated here, otherwise they need to be manually ordered
later in R. The order of prediction for the random forest model is: AlexNet -> GoogLeNet -> Inception_ResNet -> Inception_v3 
-> ResNet -> VGG

### AlexNet
```python
model_AlexNet=mx.model.FeedForward.load("/path/to/models/ARIANNA_alexnet_AMDSEV",
	epoch=20,
	ctx=ctx)
pred_results=model_AlexNet.predict(dataiter_pred)
np.savetxt("/path/to/predict/files/examples/1_AlexNet_20Epochs.txt",pred_results)
```

### GoogleNet
```python
model_GoogleNet=mx.model.FeedForward.load("/path/to/models/ARRIANNA_googlenet_AMDSEV",
	epoch=20,
	ctx=ctx)
pred_results=model_GoogleNet.predict(dataiter_pred)
np.savetxt("/path/to/predict/files/examples/2_GoogleNet_20Epochs.txt",pred_results)
```

### ResNet_inception_v2
```python
model_ResNet_inception_v2=mx.model.FeedForward.load("/path/to/models/ARIANNA_inception_resnet_v2_AMDSEV",
	epoch=20,
	ctx=ctx)
pred_results=model_ResNet_inception_v2.predict(dataiter_pred)
np.savetxt("/path/to/predict/files/examples/3_Inception_ResNet_v2_20Epochs.txt",pred_results)
```

### Inception
```python
model_Inception=mx.model.FeedForward.load("/path/to/models/ARIANNA_inception_v3_AMDSEV",
	epoch=50,
	ctx=ctx)
pred_results=model_Inception.predict(dataiter_pred)
np.savetxt("/path/to/predict/files/examples/4_Inception_v3_50Epochs.txt",pred_results)
```

### ResNet
```python
model_ResNet=mx.model.FeedForward.load("/path/to/models/ARIANNA_resnet_AMDSEV",
	epoch=20,
	ctx=ctx)
pred_results=model_ResNet.predict(dataiter_pred)
np.savetxt("/path/to/predict/files/examples/5_ResNet_20Epochs.txt",pred_results)
```

### VGG
```python
model_vgg=mx.model.FeedForward.load("/path/to/models/ARIANNA_vgg_AMDSEV",
	epoch=30,
	ctx=ctx)
pred_results=model_vgg.predict(dataiter_pred)
np.savetxt("/path/to/predict/files/examples/6_VGG_30Epochs.txt",pred_results)
```

## 3. Read the predictions from the text files and use the pre-trained random forest model to compute the final prediction

start R

```R
R
```
In the example folder, 10 images were predicted with the 6 different CNNs

```R
predict.files=list.files("/path/to/predict/files/examples/", full.names=TRUE)
```

Read all the files that contain predictions and create new matrix with results (Number of Images x 120)
```R
predict.data=do.call(cbind, lapply(predict.files, read.table))
```


Note: the CNNs were trained with 20 output nodes each since this resulted in faster convergence. The first 13 predictions (UG, AREDS classes 1-12), however, are only important for prediction in the random forest model

Name each predictor accordingly
```R
colnames(predict.data)=c(paste("AlexNet", 1:20, sep="_"), paste("GoogLeNet", 1:20, sep="_"),paste("Inception_Resnet", 1:20, sep="_"),paste("Inception", 1:20, sep="_"),paste("ResNet", 1:20, sep="_"), paste("VGG", 1:20, sep="_"))
```

Load the pre-trained random forest classifier

```R
library(randomForest)
load("/path/to/models/ARIANNA_RandomForest.RData")
```

Predict the best guess classes		

```R
prediction <- predict(fit, predict.data)
write.table(prediction)
```

Expected result for the 10 images:
> "1" -> "1"

> "2" -> "1"

> "3" -> "1"

> "4" -> "1"

> "5" -> "11"

> "6" -> "11"

> "7" -> "1"

> "8" -> "1"

> "9" -> "0"

> "10" -> "1"








