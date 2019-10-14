

#State of the ART of machine learning models to detect an segment objects 

## Introduction

**Machine learning** (**ML**) is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task.

Object segmentation is a subset of computer vision which nowadays freequently using machine learning alogrithm to deal with problems. Segmentation models are useful for a variety of tasks, including  Autonomous vehicles,Medical image diagnostics and satellite Imagery

#### Task Representing

For segementing object, essentially neural networks are widely used.

Basiclly, our goal is to segment a image ,either RGB image(height×width×3) or greyscale  image (height×width×3) and output a segmentation map pixel with a a class label represented as an integer (height×width×1). [1]![input to label](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-17-at-9.02.15-PM.png)

Similar to how we treat standard categorical values, we'll create our **target** by one-hot encoding the class labels - essentially creating an output channel for each of the possible classes.

![one hot](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-9.36.00-PM.png)

A prediction can be collapsed into a segmentation map (as shown in the first image) by taking the `argmax` of each depth-wise pixel vector.

We can easily inspect a target by overlaying it onto the observation.

![overlay](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-9.36.38-PM.png)

[1]: https://www.jeremyjordan.me/semantic-segmentation/#constructing	"An overview of semantic image segmentation."

## Architecture

In a neural network, an input diagram is decomposed into several convolutional layers, and outout a final segmental map.  this is a convenient way.However, it's quite computationally expensive to preserve the full resolution throughout the network. 

![Screen-Shot-2018-05-19-at-12.32.20-PM](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-19-at-12.32.20-PM.png)

[img 1]: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf

In deep Convolutional Neural Network, earlier layers tend to learn low-level concepts while later layers tend to have more detailed information. That means main focus shoulbe be on the latter. One popular approach for image segmentation models is to follow an **encoder/decoder structure** where we *downsample* the spatial resolution of the input, developing lower-resolution feature mappings which are learned to be highly efficient at discriminating between classes, and the *upsample* the feature representations into a full-resolution segmentation map.



## Segment Methods

#### Upsampling method

There are a few different approaches that we can use to *upsample* the resolution of a feature map. Whereas pooling operations downsample the resolution by summarizing a local area with a single value (ie. average or max pooling), "unpooling" operations upsample the resolution by distributing a single value into a higher resolution.

![Screen-Shot-2018-05-19-at-12.54.50-PM](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-19-at-12.54.50-PM.png)

[Img2]: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf	"DetectionandSegmentation"

However, **transpose convolutions** are by far the most popular approach as they allow for us to develop a *learned upsampling*.

![Screen-Shot-2018-05-19-at-3.12.51-PM](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-19-at-3.12.51-PM.png)

For filter sizes which produce an overlap in the output feature map (eg. 3x3 filter with stride 2 - as shown in the below example), the overlapping values are simply added together. Unfortunately, this tends to produce a checkerboard artifact in the output and is undesirable, so it's best to ensure that your filter size does not produce an overlap.

![padding_strides_transposed](https://www.jeremyjordan.me/content/images/2018/05/padding_strides_transposed--1-.gif)

#### Fully convolutional networks

> Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, improve on the previous best result in semantic segmentation. Our key insight is to build “fully convolutional” networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet, the VGG net, and GoogLeNet) into fully convolutional networks and transfer their learned representations by fine-tuning to the segmentation task. We then define a skip architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional networks achieve improved segmentation of PASCAL VOC (30% relative improvement to 67.2% mean IU on 2012), NYUDv2, SIFT Flow, and PASCAL-Context, while inference takes one tenth of a second for a typical image.[1]

 "fully convolutional" network trained end-to-end, pixels-to-pixels for the task of image segmentation was introduced by [Long et al.](https://arxiv.org/abs/1411.4038) in late 2014. The paper's authors propose adapting existing, well-studied *image classification*networks (eg. AlexNet) to serve as the encoder module of the network, appending a decoder module with transpose convolutional layers to upsample the coarse feature maps into a full-resolution segmentation map.

![](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-9.53.20-AM.png)

![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaTa8ut6HiawDS9XiaLq3N65EDaKWv56WCiblRrDgj5JnicrMaXp2OdUR8c8euJ6S0hTia0zJsZPibYsIN850Db7ff11A/640?wx_fmt=png)

However, the encoder decrese the resolution of the input by a fraction of 32, the decoder is hard to get a fined-grained segmentation. Other algrithm should be applied for the a higher resolution diagram.

![Screen-Shot-2018-05-20-at-10.15.09-AM](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-10.15.09-AM.png)

[img3]: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf	"Detection and Segmentation"



#### Advanced U-Net variants

The standard U-Net model consists of a series of convolution operations for each "block" in the architecture. As I discussed in my post on [common convolutional network architectures](https://www.jeremyjordan.me/semantic-segmentation/convnet-architectures/), there exist a number of more advanced "blocks" that can be substituted in for stacked convolutional layers.



> U-net architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations. [2]
>
> ![U Net](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-1.46.43-PM.png)

[Drozdzal et al.](https://arxiv.org/abs/1608.04117) swap out the basic stacked convolution blocks in favor of **residual blocks**. This residual block introduces short skip connections (within the block) alongside the existing long skip connections (between the corresponding feature maps of encoder and decoder modules) found in the standard U-Net structure. They report that the short skip connections allow for faster convergence when training and allow for deeper models to be trained.

Expanding on this, [Jegou et al.](https://arxiv.org/abs/1611.09326) proposed the use of **dense blocks**, still following a U-Net structure, arguing that the "characteristics of DenseNets make them a very good fit for semantic segmentation as they *naturally induce skip connections and multi-scale supervision*." These dense blocks are useful as they carry low level features from previous layers directly alongside higher level features from more recent layers, allowing for highly efficient feature reuse.



>  Following is the diagram of our architecture for semantic segmentation. Our architecture is built from dense blocks. The diagram is com- posed of a downsampling path with 2 Transitions Down (TD) and an upsampling path with 2 Transitions Up (TU). A circle repre- sents concatenation and arrows represent connectivity patterns in the network. Gray horizontal arrows represent skip connections, the feature maps from the downsampling path are concatenated with the corresponding feature maps in the upsampling path. Note that the connectivity pattern in the upsampling and the downsam- pling paths are different. In the downsampling path, the input to a dense block is concatenated with its output, leading to a linear growth of the number of feature maps, whereas in the upsampling path, it is not. [3]

![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaTa8ut6HiawDS9XiaLq3N65EDaKWv56WCibuqd9bdW4r6icDFwibT45zMENrAuGJkiamcUicticN5B7wqep5RDTIJ7sX1w/640?wx_fmt=png)

One very important aspect of this architecture is the fact that the upsampling path *does not* have a skip connection between the input and output of a dense block. The authors note that because the "upsampling path *increases* the feature maps spatial resolution, the linear growth in the number of features would be too memory demanding." Thus, only the *output* of a dense block is passed along in the decoder module.

![Screen-Shot-2018-05-21-at-10.44.23-PM](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-21-at-10.44.23-PM.png)

#### Dilated/atrous convolutions

> LetF :Z2 →Rbeadiscretefunction. LetΩr =[−r,r]2 ∩Z2 andletk:Ωr →Rbeadiscrete 
>
> filter of size (2r + 1)2. The discrete convolution operator ∗ can be defined as
> 											$(F∗k)(p)= 􏰂 $$\sum_{s+t=p} \frac{1}{F(s)k(t)}. $ 
>
>
>  We now generalize this operator. Let l be a dilation factor and let ∗l be defined as 
>
> ​											$(F∗k)(p)= 􏰂 $$\sum_{s+lt=p} \frac{1}{F(s)k(t)}. $ 
>
> We will refer to ∗l as a dilated convolution or an l-dilated convolution. The familiar discrete convo- lution ∗ is simply the 1-dilated convolution. [4]

One benefit of downsampling a feature map is that it *broadens the receptive field*(with respect to the input) for the following filter, given a constant filter size. Recall that this approach is more desirable than increasing the filter size due to the parameter inefficiency of large filters (discussed [here](https://arxiv.org/abs/1512.00567) in Section 3.1). However, this broader context comes at the cost of reduced spatial resolution.

**Dilated convolutions** provide alternative approach towards gaining a wide field of view while preserving the full spatial dimension. As shown in the figure below, the values used for a dilated convolution are spaced apart according to some specified *dilation rate*.

![dilation](https://www.jeremyjordan.me/content/images/2018/05/dilation.gif)

[Some architectures](https://arxiv.org/abs/1511.07122) swap out the last few pooling layers for dilated convolutions with successively higher dilation rates to maintain the same field of view while preventing loss of spatial detail. However, it is often [still too computationally expensive](https://arxiv.org/abs/1606.00915) to completely replace pooling layers with dilated convolutions.

[Img4]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md	"Convolution arithmetic"



## Reconmendation for New Learner

personally speaking, starting to learn object segmentation should follow these steps:

1. Step 0: Basics of R / Python. ... 
2. Step 1: **Learn** basic Descriptive and Inferential Statistics. ... 
3. Step 2: Data Exploration / Cleaning / Preparation. ... 
4. Step 3: Introduction to **Machine Learning**. ... 
5. Step 4: Participate in Kaggle Knowledge competition. ... 
6. Step 5: Advanced **Machine Learning**. ... 
7. Step 6: Participate in main stream Kaggle Competition.

​	

## Conclusions

to segment a image, RGB image or greyscale  image was decomposed and output as a segmentation map pixel with a a class label represented as an integer

The machine learning method for object segmentation contains Upsampling method,Fully convolutional networks with other improved method, and Dilated/atrous convolutions



## Reference 

1. Fully Convolutional Networks for Semantic Segmentation,[IEEE Transactions on Pattern Analysis and Machine Intelligence](https://ieeexplore-ieee-org.ezproxy.bu.edu/xpl/RecentIssue.jsp?punumber=34) ( Volume: 39 , [Issue: 4](https://ieeexplore-ieee-org.ezproxy.bu.edu/xpl/tocresult.jsp?isnumber=7870775) , April 1 2017 )
2. U-Net: Convolutional Networks for Biomedical Image Segmentation,**Arxiv ID:** 1505.04597
3. The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation,[arXiv:1611.09326](https://arxiv.org/abs/1611.09326) **[cs.CV]**
4. Multi-Scale Context Aggregation by Dilated Convolutions,[arXiv:1511.07122](https://arxiv.org/abs/1511.07122) **[cs.CV]**
5. Rethinking Atrous Convolution for Semantic Image Segmentation
   Evaluation of Deep Learning Strategies for Nucleus Segmentation in Fluorescence Images



