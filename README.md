# Two Layer Neural Network Classifier
## Description:

A naive implementation of convulutional neural network layers written in Python3.

## Model Details:
The layer.py includes several different layers with both forward and backward pass. Specifically, the layers include fully-connected layer, ReLU function layer, convolution layer, max pooling layer and dropout layer.

### Fully-connected layer ###
The fully-connected layer uses a simple matrix operation of 
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\inline&space;out=Wx&plus;b" title="out=Wx+b" />
</p>
where *W* is the weight matrix, *x* is the input matrix and *b* is the bias matrix. *W* typically has a shape of (*D*, *B*) with *D* as number of dimensions. *x* is usually a matrix of RBG image dataset with shape(*N*, *d*<sub>1</sub>,..., *d*<sub>c</sub>) with *N* as number of images and *d*<sub>1</sub>,..., *d*<sub>*c*</sub> as *c* channel pixel values. 




## Code Usage
This is only a naive layer implementation and is only used for understanding of CNN. It cannot be used for real dataset since it would be TOO SLOW. To vectorize the implementation, refer to [im2col](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html) to find the answer.

## Reference
1. [https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/).
2. [https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_learning.html](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_learning.html).
3. [https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/slides/L10_cnns_backprop_notes.pdf](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/slides/L10_cnns_backprop_notes.pdf)
