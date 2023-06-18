<a href="">链接文本</a>

# WeeklyAIPaperNotes

[U-Net: Convolutional Networks for Biomedical Image Segmentation(2023/5/20)](#u-net-convolutional-networks-for-biomedical-image-segmentation2023520)

[Momentum Contrast for Unsupervised Visual Representation Learning(2023/5/27)](#momentum-contrast-for-unsupervised-visual-representation-learning2023527)

[SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS(2023/6/3)](#semantic-image-segmentation-with-deep-convolutional-nets-and-fully-connected-crfs202363)

[AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE(2023/6/9)](#an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale202369)

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

+ Overlap-tile Strategy

  To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image. 

  Meanwhile, there would be a choice between larger input tiles or a larger batch size when the GPU memory is limitted.

  The authors favor large input tiles and hence reduce the batch to a single image. Accordingly they use a high momentum (0.99) such that a large number of the previously seen training samples determine the update in the current optimization step.  

+ Elastic Deformation

  It's the most common variation of the task in reality and there is very little training data available.

### Loss Fuction

<img src=".\images\image-20230520174104573.png" alt="image-20230520174104573" style="zoom:100%;" />

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

![20230617232934](D:\Desktop\images\20230617232934.png)

![20230618100821](./20230618100821.png)

Previous methods always scale up CNNs by one of the dimensions of depth/width/resolution. However, a compound scaling method is more intuitively reasonable:

The network need more layers to increase the receptive field and more channels to capture more fine-grained patterns if the input image is bigger.

However,  it requires tedious manual tuning and often yields sub-optimal accuracy and efficiency when scaling 2 or 3 dimensions arbitrarily. Thus, they try to find a principled method to scale up CNNs that can achieve better accuracy and efficiency.

### Method

+ Develop EfficientNet-B0

  Although higher accuracy is critical, we have already hit the hardware memory limit, and thus further accuracy gain needs better efficiency. Notably, the effectiveness of model scaling heavily depends on the baseline network. Considering that deep CNNs are often over-parameterized, they use a method of model compression, called neural architecture search to design a new baseline network (EfficientNet-B0).

  Neural architecture search is a popular method in designing efficient mobile-size CNNs and achieves even better efficiency than hand-crafted mobile CNNs by extensively tuning the network width, depth, convolution kernel types and sizes. However, applying this approach for large models is a huge challenge because of extremely expensive tuning cost.

  EfficientNet-B0 is scaled up to obtain a family of models, called EfficientNets. EfficientNets significantly outperform other CNNs.

![20230618001636](D:\Desktop\images\20230618001636.png)

+ Model Scaling

  + Problem Formulation
  
    A CNN is defined as $$\mathcal{N}=\bigodot_{i=1 \ldots s} \mathcal{F}_i^{L_i}\left(X_{\left\langle H_i, W_i, C_i\right\rangle}\right)$$, where $\mathcal{F}_i^{L_i}$ denotes layer $F_i$ is repeated $L_i$ times in stage $i$, $\left\langle H_i, W_i, C_i\right\rangle$ denotes the shape of input tensor $X$ of layer $i$.
  
    While regular CNN designs mostly focus on finding the best layer architecture $\mathcal{F}_i$, model scaling tries to expand the network length $\left(L_i\right)$, width $\left(C_i\right)$, and/or resolution $\left(H_i, W_i\right)$ without changing $\mathcal{F}_i$ predefined in the baseline network. 
  
    Not only all dimensions, but also all layers must be scaled uniformly in order to reduce the design space. Thus, the target is:
    $$
    \begin{aligned}
    \max_{d, w, r} & \text{Accuracy}(\mathcal{N}(d, w, r)) \\
    \text{s.t.} & \mathcal{N}(d, w, r) = \bigodot_{i=1}^{s} \hat{\mathcal{F}}_i^{d \cdot \hat{L}_i}\left(X_{\left\langle r \cdot \hat{H}_i, r \cdot \hat{W}_i, w \cdot \hat{C}_i\right\rangle}\right) \\
    & \text{Memory}(\mathcal{N}) \leq \text{target\_memory} \\
    & \text{FLOPS}(\mathcal{N}) \leq \text{target\_flops}
    \end{aligned}
    $$
    where $w, d, r$ are coefficients for scaling network width, depth, and resolution; $\hat{\mathcal{F}}_i, \hat{L}_i, \hat{H}_i, \hat{W}_i, \hat{C}_i$ are predefined parameters in baseline network.

  + Compound Scaling Method
  
    This paper proposes a new compound scaling method that uses a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a principled way:
    $$
    \begin{aligned}
    & \text { depth: } d=\alpha^\phi \\
    & \text { width: } w=\beta^\phi \\
    & \text { resolution: } r=\gamma^\phi \\
    & \text { s.t. } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \\
    & \alpha \geq 1, \beta \geq 1, \gamma \geq 1
    \end{aligned}
    $$
    where $\alpha, \beta, \gamma$ are constants that can be determined by grid search and $\phi$ is a user-specified coefficient that controls how many more resources are available for model scaling.

  + Obtain EfficientNets

    STEP 1

    Fix $\phi=1$, assuming twice more resources available, and do a small grid search of $\alpha, \beta, \gamma$. In particular, they find the best values for EfficientNet-B0 are $\alpha=1.2, \beta=$ 1.1, $\gamma=1.15$, under constraint of $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.

    STEP 2

    Fix $\alpha, \beta, \gamma$ as constants and scale up baseline network with different $\phi$ to obtain EfficientNet-B1 to B7.
      
      
  
    Note that $\alpha, \beta, \gamma$ are fixed across subsequent models instead of being re-searched for every model to avoid the extremely expensive cost.




