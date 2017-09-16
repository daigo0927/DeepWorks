# Residual Networks implemented with Keras

greatly thanks [here](http://www.iandprogram.net/entry/2016/06/06/180806)

## my main environment
- python 3.5
- tensorflow 1.1.0
- Keras 2.0.6

## usage in python code

if use preset model (18, 34, 50, 101, 152 layers)

```python
from ResNet_keras.model import ResNetBuilder

model = ResNetBuilder.build_resnet_50(input_shape = (224, 224, 3), num_output = 100)
```

of course you can directly design layer architecture
- ```block_fn``` requires residual block type, choose 'plain_block' or 'bottleneck_block'
- ```repetitions``` requires repetition number for each residual scale, input [a, b, c, d]

```python
model = ResNetBuilder.build(input_shape = (224, 224, 3), num_outputs = 100,
                            block_fn = 'plain_block', repetitions = [2, 2, 3, 3])
```
