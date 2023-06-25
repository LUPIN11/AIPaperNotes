# WeeklyAIPaperNotes

[U-Net: Convolutional Networks for Biomedical Image Segmentation(2023/5/20)](#u-net-convolutional-networks-for-biomedical-image-segmentation2023520)

[Momentum Contrast for Unsupervised Visual Representation Learning(2023/5/27)](#momentum-contrast-for-unsupervised-visual-representation-learning2023527)

[SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS(2023/6/3)](#semantic-image-segmentation-with-deep-convolutional-nets-and-fully-connected-crfs202363)

[AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE(2023/6/9)](#an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale202369)

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks(2023/6/13)](#efficientnet-rethinking-model-scaling-for-convolutional-neural-networks2023613)

[Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks(2023/6/18)](#pseudo-label-the-simple-and-efficient-semi-supervised-learning-method-for-deep-neural-networks2023618)

[TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED LEARNING(2023/6/20)](#temporal-ensembling-for-semi-supervised-learning2023620)

## U-Net: Convolutional Networks for Biomedical Image Segmentation(2023/5/20)

### Overview

This paper presents a new network architecture, U-Net, of fully convolutional network(FCN) and its training strategy to solve a pixel-level classification task with very few annotated images.

Key Points:

+ Data augmentation: overlap-tile strategy  and elastic deformation
+ Network architecture: a contracting path and a expanding path that has concatenations with the correspondingly cropped feature map from the contracting path
+ A weighted loss: weigthed cross entropy loss on each pixels, and the weight is determind by sample balancing and the distance to the border cells

### Network Architecture

<img src=".\images\image-20230520145130450.png" alt="image-20230520145130450" style="zoom:100%;" />



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

<img src=".\images\image-20230520171749916.png" alt="image-20230520171749916" style="zoom:100%;" />

Although the amount of data for medical image segmentation is very limited and U-Net has a large number of parameters owing to its larger number of channels, no regularization is mentioned in this paper (BN had not been introduced at that time). Thus, data augmentation is very crucial to avoid overfitting and achieve better generalization.

+ Overlap-tile Strategy

  To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image. 

  Meanwhile, there would be a choice between larger input tiles or a larger batch size when the GPU memory is limitted.

  The authors favor large input tiles and hence reduce the batch to a single image. Accordingly they use a high momentum (0.99) such that a large number of the previously seen training samples determine the update in the current optimization step.  

+ Elastic Deformation

  It's the most common variation of the task in reality and there is very little training data available.

### Loss Fuction

<img src=".\images\image-20230520174104573.png" alt="image-20230520174104573" style="zoom:100%;" />

Due to the challenge of separating touching objects of the same class, a weighted loss is used to force the network to learn the small separation borders.  

### Experiment Details
+ Post-processing: use the optimal threshold.
+ The output map is averaged over maps of the original data and its 7 rotated versions.


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
  
  <img src=".\images\image-20230527171548381.png" alt="image-20230527171548381" style="zoom: 100%;" />

### Implement

+ Maintain the dictionary as a queue 

  The encoded representations of the current mini-batch are enqueued, and the oldest are dequeued.

+ The key encoder is a momentum-based moving average of the query encoder

  Only the query encoder is updated by BP.

  <img src=".\images\image-20230527202252.png" alt="image-20230527202252" style="zoom: 100%;" />

Why Consistent:

Though encoders varies across each mini-batch, the difference can be made small by using a large momentum (0.999).

Why Large:

Though large, the dictionary is consistent.

Though large, the dictionary needn't to be stored on the GPU as the query encoder doesn't require gradient. 

Therefore, the momentum strategy is the core of MoCo.

  <img src=".\images\image-20230528091624.png" alt="image-20230528091624" style="zoom: 100%;" />


## SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS(2023/6/3)

### Overview

This paper proposes a fully connected CRF layer at the end of DCNNs to better localize segment boundaries.

Key Points:

+ Atrous convolution and network re-purposing
+ Fully connected CRF for accurate localization

### Previous Bottleneck

+ spatial invariance
+ signal downsampling

The success of DCNNs can be partially attributed to its build-in invariance to local image transformations, which enables them to learn more abstract representations. While this invariance is desirable for high-level vision tasks, it can hamper low-level tasks because of the lack of precise localization. 

Meanwile, repeatedly signal downsampling, including max-pooling and striding, results in a decrease in resolution. These two problems are the main hurdles in applying DCNNs to dense image labeling.

This paper tries to address the invariance problem using CRF and the downsampling problem using atrous convolution.

### Implement
<img src=".\images\20230603141152.png" alt="20230603141152" style="zoom:100%;" />

The first step is to obtain the dense score maps output by a DCNN. For efficiency, network re-purposing is employed on a pre-trained model (VGG16).  Then, thanks to the smoothness of the score maps, their resolution can be restored by simple bilinear interpolation.

Note that atrous convolution plays an indispensable role in this step.

+ Efficient computing

  With the same number of parameters and computational cost, atrous convolution provides a larger receptive field.

+ Better downsampling

  It allows arbitrary downsampling rates without introducing any approximations.

+ Smooth score maps

  Otherwise, learning upsampling layers would significantly increase the complexity and training time.

  <img src=".\images\20230603152340.png" alt="20230603152340" style="zoom:100%;" />

Last, couple the recognition capacity of DCNNs and the fine-grained localization accuracy of fully connected CRFs.

  <img src=".\images\20230603141325.png" alt="20230603141325" style="zoom:100%;" />

#### Fully Connected CRFs
Note that there exists a dependency between any two pixels in the fully connected CRF layer.

The energy function of the fully connected CRF layer is:

$$
E(x) = \sum_{i}\theta_i(x_i) + \sum_{ij}\theta_{ij}(x_i, x_j)
$$

$$
\theta_i(x_i)=-logP(x_i)
$$

$$
\theta_{ij}=\mu(x_i, x_j)\sum_{m=1}^{K}\omega_m\cdot{k^m(f_i,f_j)}
$$

where $i,j$  stand for pixel $i$ and pixel $j$, $P(x_i)$ is the label assignment probability computed by the DCNN at pixel $i$, $\mu(x_i, x_j)=1$ if $x_i \not= x_j$, and 0 otherwise, $f_i$ stands for features extracted for pixel $i$ and $k^m$ is a Gaussian kernel. 

The kernels are:

$$
\omega_1\cdot{exp(-\frac{||p_i-p_j||^2}{2\sigma_\alpha^2}-\frac{||I_i-I_j||^2}{2\sigma_\beta^2}) +\omega_2\cdot{exp(-\frac{||p_i-p_j||^2}{2\sigma_\gamma^2})}}
$$

where $p$ stands for pixel position and $I$ stands for pixel color intensity. 


## AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE(2023/6/9)  
### Overview

This paper shows that vision transformer(ViT) outperforms CNNs on many downstream tasks when pre-trained on large amounts of data.  
Note that there are many insightful experiments in this paper.

### Implement

<img src=".\images\20230609194323.png" alt="20230609194323" style="zoom:100%;" />

+ Reshape image $x \in \mathbb{R}^{H \times W \times C}$ into $x_p \in \mathbb{R}^{N \times (P^2\cdot C)}$, where $(P,P)$ is the resolution of each image patch. In other words, each patch ($\mathbb{R}^{P \times P \times C}$) is flattened.
+ Map these flattened patches to $D$ dimensions with a trainable linear projection. The output of this projection are patch embeddings.
+ $z^0_{0}$ is a learnable embedding and the corresponding output, $z_L^{0}$, is the image representation $y$ produced by Transformer Encoder.
+ Use standard learnable 1D position embeddings since there are no significant performance gains when using more advanced 2D-aware position embeddings.

### Conclusions

+ It become possible to train models of unprecedented size (e.g. 100B) thanks to the efficiency and scalability of Transformers. And there is still no sign of saturating performance.
+ Since Transformers has much less image-specific inductive bias than CNNs, it do not generalize well when trained on insufficient amounts of data. But large scale training trumps inductive bias.
+ It is ofen beneficial to fine-tune at higher resolution than pre-training since a higher resolution results in a larger effective sequence length.


## EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks(2023/6/13)

This paper proposes a new method to scale up CNNs to achieve better performance. The method uniformly scales up all dimensions of depth/width/resolution using a compound coefficient. Furthermore, they design a new baseline network using neutral achitecture search and scale it up to obtain a family of models, called EfficientNets.

Key Points:

+ A highly effective method to scale up CNNs
+ A new family of models, called EfficientNets

### Motivation

<img src=".\images\20230617232934.png" alt="20230617232934" style="zoom:100%;" />

<img src=".\images\20230618100821.png" alt="20230618100821" style="zoom:100%;" />

Previous methods always scale up CNNs by one of the dimensions of depth/width/resolution. However, a compound scaling method is more intuitively reasonable:

The network need more layers to increase the receptive field and more channels to capture more fine-grained patterns if the input image is bigger.

However,  it requires tedious manual tuning and often yields sub-optimal accuracy and efficiency when scaling 2 or 3 dimensions arbitrarily. Thus, they try to find a principled method to scale up CNNs that can achieve better accuracy and efficiency.

### Method

+ Develop EfficientNet-B0

  Although higher accuracy is critical, we have already hit the hardware memory limit, and thus further accuracy gain needs better efficiency. Notably, the effectiveness of model scaling heavily depends on the baseline network. Considering that deep CNNs are often over-parameterized, they use a method of model compression, called neural architecture search to design a new baseline network (EfficientNet-B0).

  Neural architecture search is a popular method in designing efficient mobile-size CNNs and achieves even better efficiency than hand-crafted mobile CNNs by extensively tuning the network width, depth, convolution kernel types and sizes. However, applying this approach for large models is a huge challenge because of extremely expensive tuning cost.

  EfficientNet-B0 is scaled up to obtain a family of models, called EfficientNets. EfficientNets significantly outperform other CNNs.

<img src=".\images\20230618001636.png" alt="20230618001636" style="zoom:100%;" />


 + Problem Formulation
  
    A CNN is defined as $`\mathcal{N}=\bigodot_{i=1 \ldots s} \mathcal{F}_i^{L_i}\left(X_{\left\langle H_i, W_i, C_i\right\rangle}\right)`$, where $\mathcal{F}_i^{L_i}$ denotes layer $F_i$ is repeated $L_i$ times in stage $i$, $\left\langle H_i, W_i, C_i\right\rangle$ denotes the shape of input tensor $X$ of layer $i$.
  
    While regular CNN designs mostly focus on finding the best layer architecture $\mathcal{F}_i$, model scaling tries to expand the network length $\left(L_i\right)$, width $\left(C_i\right)$, and/or resolution $\left(H_i, W_i\right)$ without changing $\mathcal{F}_i$ predefined in the baseline network. 
  
    Not only all dimensions, but also all layers must be scaled uniformly in order to reduce the design space. Thus, the target is: [see formulas below], where $w, d, r$ are coefficients for scaling network width, depth, and resolution; $\hat{\mathcal{F}}_i, \hat{L}_i, \hat{H}_i, \hat{W}_i, \hat{C}_i$ are predefined parameters in baseline network.
```math
  \begin{aligned}
  \max_{d, w, r} & \text{Accuracy}(\mathcal{N}(d, w, r)) \\
  \text{s.t.} & \mathcal{N}(d, w, r) = \bigodot_{i=1}^{s} \hat{\mathcal{F}}_i^{d \cdot \hat{L}_i}\left(X_{\left\langle r \cdot \hat{H}_i, r \cdot \hat{W}_i, w \cdot \hat{C}_i\right\rangle}\right) \\
  & \text{Memory}(\mathcal{N}) \leq \text{target\_memory} \\
  & \text{FLOPS}(\mathcal{N}) \leq \text{target\_flops}
  \end{aligned}
```
  
  + Compound Scaling Method
  
    This paper proposes a new compound scaling method that uses a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a principled way: [see formulas below], where $\alpha, \beta, \gamma$ are constants that can be determined by grid search and $\phi$ is a user-specified coefficient that controls how many more resources are available for model scaling.
```math
  \begin{aligned}
  & \text { depth: } d=\alpha^\phi \\
  & \text { width: } w=\beta^\phi \\
  & \text { resolution: } r=\gamma^\phi \\
  & \text { s.t. } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \\
  & \alpha \geq 1, \beta \geq 1, \gamma \geq 1
  \end{aligned}
``` 
  + Obtain EfficientNets
  
    - STEP 1
  
      Fix $\phi=1$, assuming twice more resources available, and do a small grid search of $\alpha, \beta, \gamma$. In particular, they find the best values for EfficientNet-B0 are $\alpha=1.2, \beta=$ 1.1, $\gamma=1.15$, under constraint of $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.
  
    - STEP 2
  
      Fix $\alpha, \beta, \gamma$ as constants and scale up baseline network with different $\phi$ to obtain EfficientNet-B1 to B7.
  
    Note that $\alpha, \beta, \gamma$ are fixed across subsequent models instead of being re-searched for every model to avoid the extremely expensive cost.

## Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks(2023/6/18)
### Overview 

This paper proposes a method of semi-supervised learning, called Pseudo-Label. The network is trained in a supervised fashion with labeled samples and unlabled samples that treat pseudo-labels as true labels.

### Implement

Like all of recent successful methods for training deep neural networks, there are two phrase of training: unsupervised pre-training and fine-tuning. Pseudo-Labels are used in the second phrase. The pre-trained network is trained in a supervised fashion with labeled and unlabeled data simultaneously. 

Pseudo-Labels are the classes that have maximum predicted probabilities for each unlabeled sample and they will be treated as if they were true labels. Note that pseudo-labels are re-calculated every weights update. The overall loss fuction is: 

```math
L=\frac{1}{n} \sum_{m=1}^n \sum_{i=1}^C L\left(y_i^m, f_i^m\right)+\alpha(t) \frac{1}{n^{\prime}} \sum_{m=1}^{n^{\prime}} \sum_{i=1}^C L\left(y_i^{\prime m}, f_i^{\prime m}\right)
```

where n is the number of labeled data in a mini-batch, $n^{'}$ for unlabeled data and $y^{'m}_i$ is the pseudo-label of unlabeled data. $\alpha(t)$ is a coefficient to balancing the loss of these two kinds of data. 

The proper scheduling of $`\alpha(t)`$ is very important for the network performance.  If $`\alpha(t)`$ is too high, it disturbs training even for labeled data. Whereas if $`\alpha(t)`$ is too small, it prevents the benefit from unlabeled data. Furthermore, $`\alpha(t)`$ should be slowly increased. This can help the model to avoid poor local minima. Note that unlabeled data is not used for training during the first $`T_i`$ iterations.

$$
\alpha(t) = 
\begin{cases}
0 & \text{if } t < T_1 \\
\frac{t - T_1}{T_2 - T_1} \alpha_f & \text{if } T_1 \leq t < T_2 \\
\alpha_f & \text{if } T_2 \leq t \\
\end{cases}
$$

### Explanation

The second term of the loss fuction is equivalent to minimizing the conditional entropy of class probabilities for unlabeled data.
```math
H\left(y \mid x^{\prime}\right)=-\frac{1}{n^{\prime}} \sum_{m=1}^{n^{\prime}} \sum_{i=1}^C P\left(y_i^m=1 \mid x^{\prime m}\right) \log P\left(y_i^m=1 \mid x^{\prime m}\right)
```
This entropy is a measure of class overlap. As class overlap decreases, the density of data points get lower at the decision boundary.  And according to the cluster assumption, the decision boundary should lie in low-density regions to improve generalization performance. Thus, pseudo-labels help the model to get better generalization performance.

<img src=".\images\20230619025252.png" alt="20230619025252" style="zoom: 67%;" />

### Experiments

<img src=".\images\20230619025357.png" alt="20230619025357" style="zoom: 67%;" />

Note that using Denoising Auto-Encoder when pre-training boosts up the performance.  


## TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED LEARNING(2023/6/20)

### Overview

This paper proposes self-ensembling, a method that can also be utilized in a semi-supervised fashion.

Key Points:

+ Self-ensembling: ensemble predictions of different epochs, regularization and augmentation contiditions
+ Semi-supervised learning: encourage these predictions, actually the entire output vectors, to be consistent.

### Introduction

A single network is trained in self-ensembling, but these emsembled predictions of different epochs and different regularization and augmentation conditions can be considered as predictions of a large number of individual sub-networks. The diversity among these sub-networks comes from:

+ Dropout: the complete network can be seen as an implicit ensemble of sub-networks
+ Augmentation:  the input augmentation are versatile and stochastic
+ The network parameters are updated every mini-batch

Notice that dropout regularization and versatile input augmentation are very crucial.

Compared with the current output, the ensemble predictions are likely to be closer to the correct but unknown labels of these unlabeled data, allowing for a better performance of semi-supervised learning.

### Methods

<img src=".\images\20230620184935.png" alt="20230620184935" style="zoom:100%;" />

Specifically, self-ensembling is achieved by calculating the weighted moving average of preditions and semi-supervised learning is achieved by minimizing the mean squared error between the entire output vectors, unlike Pseudo-Label.

Specifically, self-ensembling is achieved through moving average, and self-supervision is implemented by minimizing the mean squared error of the outputs.

<img src=".\images\20230620202903.png" alt="20230620202903" style="zoom:100%;" />

<img src=".\images\20230620202914.png" alt="20230620202914" style="zoom:100%;" />

The unsupervised loss term is scaled by a time-dependent weighting function $w(t)$, which ramps up from zero.  And it is very important that the ramp-up of the unsupervised loss term is slow enoungh-otherwise, the model gets easily stuck in a degenerate solution where no meaningful classification of the data is obtained.  

Experiments show that temporal ensembling ahieves better performance than $\Pi$-model and it is nearly 2x faster (but it takes additional space to store the ensemble predictions, especially when the dataset and number of classes are pretty large).

Notice that the training targets $\hat{z}$ are obtained by dividing $Z$ by $(1-\alpha^t)$. This step is known as bias correction, which is also employed in the Adam optimizer. The reason for bias correction is that the initial value of $Z$ is zero which leads to an underestimation of the value of $Z$. At the t-th step, the weight for the initial value of zero is $\alpha^t$. By removing this weight, we obtain a total weight of $(1-\alpha^t)$, which is used to scale the value of $Z$ to achieve bias correction.


