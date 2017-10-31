# U-Net implemented with tensorflow

[original paper](https://arxiv.org/abs/1505.04597)

## my main environment
- python 3.5
- tensorflow 1.1.0

## description
The name 'U'-Net is derived from its shape, see figure in the original paper. Though the purpose of the paper was biomedical image segmentation, this network can be utilize in many task (like style tranfering, super resolution, segmentation, etc...).

## usage in python code
Coded model expect the same size (height, width) between input and output. You just design output channel.

```python
from U_Net.model import U_Net

model = U_Net(ouput_ch = 10, block_fn = 'origin')
```

- ```output_ch``` is number of output channel, requires int value.
- ```block_fn``` requires type of convolution block, choose 'origin' or 'batch_norm'
  - 'origin' is (I think) same as original paper.
  - 'batch_norm' use BatchNormalization.

```python
import tensorflow as tf

# utilize model
inputs = tf.placeholder(tf.float32, shape = (None, 128, 128, 3))
outputs_ = model(inputs, reuse = False)
# script wil continues...
```
