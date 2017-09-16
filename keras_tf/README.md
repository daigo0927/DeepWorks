# Keras and tensorflow combination
Simple CNN implementation with keras and tensorflow, I wish this to be a milestone for easy and flexible DeepLearning implementation.

Detailed article is [here](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html).

## my main environment
- python 3.5
- tensorflow 1.1.0
- Keras 2.0.6

## description

- Model is coded with keras, optimization is coded with tensorflow.
- Model weights can be saved with keras interface ```model.save_weights('path/to/weights.h5')```, 
and also loaded ```model.load_weights('/path/to/weights.h5')```.
- Learned weghts with tensorflow interface may not work in keras inference interface ```model.predict(x)```,
better to use ```tf.Session()``` (want someone to confirm).

## usage
- train (set epochs:20, batchsize:128 as default)

```
python main.py
```

- validation of learned weights

```
python main.py -w /path/to/weights.h5
```
