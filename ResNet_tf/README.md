# Residual Networks implemented with tensorflow

greatly thanks [here](http://www.iandprogram.net/entry/2016/06/06/180806)

## my main environment
- python 3.5
- tensorflow 1.1.0

## usage in python code

if use preset model (18, 34, 50, 101, 152, 200 layers)

```python
from ResNet_tf.model import ResNetBuilder

model = ResNetBuilder.build_resnet50(num_output = 100)
```

of course you can directly design layer architecture
- ```block_fn``` requires residual block type, choose 'plain_block' or 'bottleneck_block'
- ```repetitions``` requires repetition number for each residual scale, input [a, b, c, d]

```python
model = ResNetBuilder.build(num_outputs = 100,
                            block_fn = 'plain', repetitions = [2, 2, 3, 3])
```
