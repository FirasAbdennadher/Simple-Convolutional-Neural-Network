# Two Layer Neural Network Classifier
## Description:

A naive implementation of convulutional neural network layers written in Python3.

## Model Details:
The layer.py includes several different layers with both forward and backward pass. Specifically, the layers include fully-connected layer, ReLU function layer, convolution layer, max pooling layer and dropout layer.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?V_t=-\frac{\alpha}{\delta&space;&plus;V_t}\triangledown_wL(W,x,&space;y)" title="V_t=-\frac{\alpha}{\delta +V_t}\triangledown_wL(W,x, y)" />
</p>

where <img src="https://latex.codecogs.com/gif.latex?\rho" title="\rho" /> is the decay rate, <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> the learning rate, <img src="https://latex.codecogs.com/gif.latex?\delta" title="\delta" /> a small constant (10<sup>-6</sup> for example), and <img src="https://latex.codecogs.com/gif.latex?\triangledown&space;L" title="\triangledown L" /> the calculated gradient.

## Code Usage
This is only a naive layer implementation and is only used for understanding of CNN. It cannot be used for real dataset since it would be TOO SLOW. To vectorize the implementation, refer to [im2col](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html) to find the answer.

## Reference
1. [https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/).
2. [https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_learning.html](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_learning.html).
3. [https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/slides/L10_cnns_backprop_notes.pdf](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/slides/L10_cnns_backprop_notes.pdf)
