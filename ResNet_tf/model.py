import tensorflow as tf
import tensorflow.contrib.layers as tcl

def _bn_relu_conv(filters, kernel_size = (3, 3), stride = (1, 1)):
    def f(inputs):
        x = tcl.batch_norm(inputs)
        x = tf.nn.relu(x)
        x = tcl.conv2d(x,
                       num_outputs = filters,
                       kernel_size = kernel_size,
                       stride = stride,
                       padding = 'SAME')
        return x
    return f

def _shortcut(inputs, x): # x = f(inputs)
    # shortcut path
    _, inputs_h, inputs_w, inputs_ch = inputs.shape.as_list()
    _, x_h, x_w, x_ch = x.shape.as_list()
    stride_h = int(round(inputs_h / x_h))
    stride_w = int(round(inputs_w / x_w))
    equal_ch = inputs_ch == x_ch

    if stride_h>1 or stride_w>1 or not equal_ch:
        shortcut = tcl.conv2d(inputs,
                              num_outputs = x_ch,
                              kernel_size = (1, 1),
                              stride = (stride_h, stride_w),
                              padding = 'VALID')
    else:
        shortcut = inputs

    merged = tf.add(shortcut, x)
    return merged

def plain_block(filters, subsample = (1, 1)):
    def f(inputs):
        x = _bn_relu_conv(filters, stride = subsample)(inputs)
        x = _bn_relu_conv(filters)(x)

        return _shortcut(inputs, x)
    return f

def bottleneck_block(filters, subsample = (1, 1)):
    def f(inputs):
        x = _bn_relu_conv(filters, kernel_size = (1, 1),
                          stride = subsample)(inputs)
        x = _bn_relu_conv(filters, kernel_size = (3, 3))(x)
        x = _bn_relu_conv(filters, kernel_size = (1, 1))(x)

        return _shortcut(inputs, x)
    return f

def _residual_block(block_fn, filters, repetition, is_first_layer = False):
    def f(inputs):
        x = inputs
        for i in range(repetition):
            subsample = (1, 1)
            if i == 0 and not is_first_layer:
                subsample = (2, 2)
            x = block_fn(filters, subsample)(x)
        return x
    return f

class ResNet(object):
    def __init__(self,
                 num_output,
                 block_fn,
                 repetitions):
        self.num_output = num_output
        self.block_fn = block_fn
        self.repetitions = repetitions
        self.name = 'resnet'

    def __call__(self, inputs, reuse = True):
        with tf.variable_scope(self.name) as vs:
            tf.get_variable_scope()
            if reuse:
                vs.reuse_variables()

            conv1 = tcl.conv2d(inputs,
                               num_outputs = 64,
                               kernel_size = (7, 7),
                               stride = (2, 2),
                               padding = 'SAME')
            conv1 = tcl.batch_norm(conv1)
            conv1 = tf.nn.relu(conv1)
            conv1 = tcl.max_pool2d(conv1,
                                   kernel_size = (3, 3),
                                   stride = (2, 2),
                                   padding = 'SAME')
            
            x = conv1
            filters = 64
            first_layer = True
            for i, r in enumerate(self.repetitions):
                x = _residual_block(self.block_fn,
                                    filters = filters,
                                    repetition = r,
                                    is_first_layer = first_layer)(x)
                filters *= 2
                if first_layer:
                    first_layer = False

            _, h, w, ch = x.shape.as_list()
            outputs = tcl.avg_pool2d(x,
                                     kernel_size = (h, w),
                                     stride = (1, 1))
            outputs = tcl.flatten(outputs)
            logits = tcl.fully_connected(outputs, num_outputs = self.num_output,
                                         activation_fn = None)
            return logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class ResNetBuilder(object):

    @staticmethod
    def build(num_output,
              block_fn,
              repetitions):

        assert block_fn in ['plain', 'bottleneck'], 'choose \'plain\' or \'bottleneck\''
        if block_fn == 'plain':
            block_fn = plain_block
        elif block_fn == 'bottleneck':
            block_fn = bottleneck_block
        
        model = ResNet(num_output = num_output,
                       block_fn = block_fn,
                       repetitions = repetitions)
        return model

    @staticmethod
    def build_resnet18(num_output):
        return ResNetBuilder.build(num_output = num_output,
                                   block_fn = 'plain',
                                   repetitions = [2, 2, 2, 2])
    @staticmethod
    def build_resnet34(num_output):
        return ResNetBuilder.build(num_output = num_output,
                                   block_fn = 'plain',
                                   repetitions = [3, 4, 6, 3])
    @staticmethod
    def build_resnet50(num_output):
        return ResNetBuilder.build(num_output = num_output,
                                   block_fn = 'bottleneck',
                                   repetitions = [3, 4, 6, 3])
    @staticmethod
    def build_resnet101(num_output):
        return ResNetBuilder.build(num_output = num_output,
                                   block_fn = 'bottleneck',
                                   repetitions = [3, 4, 23, 3])
    @staticmethod
    def build_resnet152(num_output):
        return ResNetBuilder.build(num_output = num_output,
                                   block_fn = 'bottleneck',
                                   repetitions = [3, 8, 36, 3])
    @staticmethod
    def build_resnet200(num_output):
        return ResNetBuilder.build(num_output = num_output,
                                   block_fn = 'bottleneck',
                                   repetitions = [3, 24, 36, 3])
