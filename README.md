# WeeklyAIPaperNotes

[U-Net: Convolutional Networks for Biomedical Image Segmentation(2023/5/20)](#u-net-convolutional-networks-for-biomedical-image-segmentation2023520)

[Momentum Contrast for Unsupervised Visual Representation Learning(2023/5/27)](#momentum-contrast-for-unsupervised-visual-representation-learning2023527)

## U-Net: Convolutional Networks for Biomedical Image Segmentation(2023/5/20)

Paper Download Link:

> https://arxiv.org/abs/1505.04597

### Overview

This paper presents a new network architecture, U-Net, of fully convolutional network(FCN) and its training strategy to solve a pixel-level classification task with very few annotated images.

Key Points:

+ Data augmentation: overlap-tile strategy  and elastic deformation
+ Network architecture: a contracting path and a expanding path that has concatenations with the correspondingly cropped feature map from the contracting path
+ A weighted loss: weigthed cross entropy loss on each pixels, and the weight is determind by sample balancing and the distance to the border cells

### Network Architecture

<img src=".\images\image-20230520145130450.png" alt="image-20230520145130450" style="zoom:67%;" />



The network has a contracting path to capture context by successive convolutional operations and a expanding path that enables precise localization by up-convolution and cropping from the contracting path.

At the final layer, a 1x1 convolution is used to map channel features to the desire number of classes.

Main Ideas:

+ Replacing unpooling with up-convolution to increase the resolution

  Up-convolution is trainable, whereas unpooling is not.

+ Cropping high resolution features from the contracting path to localize

  It's a concatenation operation rather than a residual connection that always involves an add operation.

+ Larger number of feature channels when upsampling to propagate context information

  Generally, we double(halve) the number of feature channels after a 2x2 max-pooling(up-convolution) operation to stabilize the number of features. However, we noticed that the number of channels remains unchanged before and after the up-convolution operation. 

### Data Augmentation

<img src=".\images\image-20230520171749916.png" alt="image-20230520171749916" style="zoom:50%;" />

+ Overlap-tile Strategy

  To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image. 

  Meanwhile, there would be a choice between larger input tiles or a larger batch size when the GPU memory is limitted.

  The authors favor large input tiles and hence reduce the batch to a single image. Accordingly they use a high momentum (0.99) such that a large number of the previously seen training samples determine the update in the current optimization step.  

+ Elastic Deformation

  It's the most common variation of the task in reality and there is very little training data available.

### Loss Fuction

<img src=".\images\image-20230520174104573.png" alt="image-20230520174104573" style="zoom:67%;" />

Due to the challenge of separating touching objects of the same class, a weighted loss is used to force the network to learn the small separation borders.  


## Momentum Contrast for Unsupervised Visual Representation Learning(2023/5/27)

### Overview

This paper presents MoCo, a contrastive unsupervised representation learning mechanism.

MoCo can achieve comparable or even better performance compared to its supervised pre-training counterpart and requires lower GPU memory.

Key Points:

+ View contrastive learning as dictionary look-up
+ Maintain the dictionary of encoded representations as a queue of mini-batches
+ A moving-averaged key encoder

### Previous Bottleneck

Dictionary building may be the key to the success of unsupervised representation learning in CV:

+ Unsupervised learning in NLP is highly sucessful and it's partly based on tokenized dictionaries.
+ In contrast, building a dictionary is more challenging in CV because the raw signal is:
  + continuous  
  + high-dimensional  
  + not structured for human communication

Contrastive unsupervised learning can be viewed as building dynamic dictionaries: 

An encoded query should be similar to its matching key (the positive sample) and dissimilar to others (negative samples).  

However, dictionaries built by previous mechanisms are limited in one of these two aspects:

+ Large

  A larger dictionary may better sample the underlying visual space.

+ Consistent

  Keys in the dictionary should be represented by the same or similar encoder so that their comparisons to the query are consistent.
  
  <img src=".\images\image-20230527171548381.png" alt="image-20230527171548381" style="zoom: 80%;" />

### Implement

+ Maintain the dictionary as a queue 

  The encoded representations of the current mini-batch are enqueued, and the oldest are dequeued.

+ The key encoder is a momentum-based moving average of the query encoder

  Only the query encoder is updated by BP.

  <img src=".\images\image-20230527202252.png" alt="20230527202252" style="zoom: 80%;" />

Why Consistent:

Though encoders varies across each mini-batch, the difference can be made small by using a large momentum (0.999).

Why Large:

Though large, the dictionary is consistent.

Though large, the dictionary needn't to be stored on the GPU as the query encoder doesn't require gradient. 

Therefore, the momentum strategy is the core of MoCo.










