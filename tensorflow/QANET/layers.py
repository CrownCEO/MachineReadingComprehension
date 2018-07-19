import tensorflow as tf

'''
Some functions are taken directly from Tensor2Tensor Library:
https://github.com/tensorflow/tensor2tensor/
and BiDAF repository:
https://github.com/allenai/bi-att-flow
'''
# 三种权重的初始化方法 Gaussian Xavier MSRA https://blog.csdn.net/qq_26898461/article/details/50996507
# 通过使用Xavier这种初始化方法，我们能够保证输入变量的变化尺度不变，从而避免变化尺度在最后一层网络中爆炸或者弥散
initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True,
                                                                     dtype=tf.float32)

initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,
                                                                          dtype=tf.float32)

# 对weight施加正则化率为regularization_rate的L2正则化（L1同理）。注意，两个括号之间没有逗号（，）。函数返回一个施加了正则化项的tensor，加入loss中
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


# tensorflow中的tf.nn.conv2d函数，实际上相当于用filter，以一定的步长stride在image上进行滑动，计算重叠部分的内积和，即为卷积结果
# input的shape: [batch, in_height, in_width, in_channels]
# filter的shape: [filter_height, filter_width, in_channels, out_channels]
def conv(inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        # VALID‘方式表示不进行扩展; ‘SAME‘表示添加0 指的是padding 添加0
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_", filter_shape, dtype=tf.float32, regularizer=regularizer,
                                  initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_", bias_shape, regularizer=regularizer, initializer=tf.zeros_initializer())

        if activation is not None:
            return activation(outputs)
        else:
            return outputs


# highway Networks就是一种解决深层次网络训练困难的网络框架
# 参看文章 highway networks https://arxiv.org/pdf/1505.00387.pdf
# https://blog.csdn.net/sinat_35218236/article/details/73826203
def highway(x, size=None, activation=None, num_layers=2, scope="highway", dropout=0.0, reuse=None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            # -1 表示从倒数第二个开始往前所有的
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name="input_projection", reuse=reuse)
        for i in range(num_layers):
            T = conv(x, size, bias=True, activation=tf.sigmoid, name="gate_%d" % i, reuse=reuse)
            H = conv(x, size, bias=True, activation=activation, name="activation_%d" % i, reuse=reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x


def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, mask=None, num_filters=128,input_projection=False, num_heads=8,
                   seq_len=None, scope="res_block", is_traing=True, reuse=None, bias=True, dropout=0.0):
    with tf.variable_scope(scope, reuse=reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name="input_projection", reuse=reuse)
        outputs = inputs
        sublayer = 1
        total_sublayers = (num_conv_layers + 2) * num_blocks
        for i in range(num_blocks):
            outputs = add_timing_signal_1d(outputs)
            outputs, sublayer = conv_block(outputs, num_conv_layers, kernel_size, num_filters, seq_len=seq_len,
                                           scope="encoder_block_%d" % i, reuse=reuse, bias=bias, drop_out=dropout, is_training=is_traing)



