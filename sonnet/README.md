# CNN with Sonnet
Simple CNN implementation with [Sonnet](https://github.com/deepmind/sonnet)

## my main environment
- python 3.5
- tensorflow 1.1.0
- dm-sonnet 1.13

## description

- Sonnet is wrapper of tensorflow

## usage
- train (set epochs:20, batchsize:128 as default)

```
python train.py
```

- validation of learned weights

```
python train.py -w /path/to/weights.h5
```
