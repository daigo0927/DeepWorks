import tensorflow as tf
import tensorflow.contrib.layers as tcl

# subpixel CNN layer (proposed in subpixel: A subpixel convolutional neural netowrk)

def PhaseShift(x, r): # x: input tensor, r: magnification value
    bsize, h, w, ch = x.shape.as_list()
    assert ch%(r**2) == 0, 'input channel should be multiplies of r^2'

    x = tf.reshape(x, (bsize, h, w, r, r, -1)) 
    x = tf.transpose(x, (0, 1, 2, 4, 3, 5)) # shape(bsize, h, w, r, r, new_ch)
    x = tf.split(x, h, axis = 1) # len(x):h, each shape(bsize, 1, w, r, r, new_ch)
    x = tf.concat([tf.squeeze(x_) for x_ in x], axis = 2) # shape(bsize, w, h*r, r, new_ch)
    x = tf.split(x, w, axis = 1) # len(x):w, each shape(bsize, 1, h*r, r, new_ch)
    x = tf.concat([tf.squeeze(x_) for x_ in x], axis = 2) # shape(bsize, h*r, w*r, new_ch)

    return x

def PhaseShift_withConv(x, r, filters, kernel_size = (3, 3), stride = (1, 1)):
    # output shape(batch, r*x_h, r*x_w, filters)

    x = tcl.conv2d(x,
                   num_outputs = filters*r**2,
                   kernel_size = kernel_size,
                   stride = stride,
                   padding = 'SAME')
    x = PhaseShift(x, r)
    return x
