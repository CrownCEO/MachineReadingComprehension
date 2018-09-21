import tensorflow as tf
import math
from operator import mul
from functools import reduce
from tensorflow.python.ops.rnn import dynamic_rnn as _dynamic_rnn, \
    bidirectional_dynamic_rnn as _bidirectional_dynamic_rnn

from tensorflow.python.ops import nn_ops

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
# 在外面调用 传递进来的inputs.shape= [N * c_maxlen, 16, 64] output_size = 96
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
            # kernel_size =5 shapes[-1]=64 output_size=96
            # [5,64,96]
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        # VALID‘方式表示不进行扩展; ‘SAME‘表示添加0 指的是padding 添加0
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_", filter_shape, dtype=tf.float32, regularizer=regularizer,
                                  initializer=initializer_relu() if activation is not None else initializer())
        # outputs shape: [N * c_maxlen, (16 - 5 + 1), 96] 等再看看
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
# 在此处size=96
def highway(x, size=None, activation=None, num_layers=2, scope="highway", dropout=0.0, reuse=None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            # -1 表示取最后一个
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name="input_projection", reuse=reuse)
        for i in range(num_layers):
            T = conv(x, size, bias=True, activation=tf.sigmoid, name="gate_%d" % i, reuse=reuse)
            H = conv(x, size, bias=True, activation=activation, name="activation_%d" % i, reuse=reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x


# 双向RNN
def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None, initial_state_fw=None,
                              initial_state_bw=None, dtype=None, parallel_iterations=None, swap_memory=False,
                              time_major=False, scope=None):
    assert not time_major
    flat_inputs = flatten(inputs, 2)
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')
    (flat_fw_outputs, flat_bw_outputs), final_state = _bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs,
                                                                                 sequence_length=flat_len,
                                                                                 initial_state_fw=initial_state_fw,
                                                                                 initial_state_bw=initial_state_bw,
                                                                                 dtype=dtype,
                                                                                 parallel_iterations=parallel_iterations,
                                                                                 swap_memory=swap_memory,
                                                                                 time_major=time_major, scope=scope)
    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    return (fw_outputs, bw_outputs), final_state


# 计算相似性矩阵
def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, input_keep_prob=1.0,
                                      scope='efficient_trilinear',
                                      bias_initializer=tf.zeros_initializer(),
                                      kernel_initializer=initializer()):
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):
        weights4arg0 = tf.get_variable(
            "linear_kernel4arg0", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4arg1 = tf.get_variable(
            "linear_kernel4arg1", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4mlu = tf.get_variable(
            "linear_kernel4mul", [1, 1, arg_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        biases = tf.get_variable(
            "linear_bias", [1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer)
        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
        subres1 = tf.tile(tf.transpose(dot(droped_args[1], weights4arg1), perm=(0, 2, 1)), [1, c_maxlen, 1])
        subres2 = batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm=(0, 2, 1)))
        res = subres0 + subres1 + subres2
        nn_ops.bias_add(res, biases)
        return res


# 多维度点积
# 只需a矩阵的最后一维dim等于b矩阵倒数第二维dim即可，对应二维情况就是第一个的列数等于第二个矩阵行数
# 也就是说点积发生在a,b矩阵最后两个维度上
def dot(x, y):
    """Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.
    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out


def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.
    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.
    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.
    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * diff], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * diff], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out


def ndim(x):
    """Copied from keras==2.0.6
    Returns the number of axes in a tensor, as an integer.
    # Arguments
        x: Tensor or variable.
    # Returns
        Integer (scalar), number of axes.
    # Examples
    ```python
        #>>> from keras import backend as K
        #>>> inputs = K.placeholder(shape=(2, 4, 5))
        #>>> val = np.array([[1, 2], [3, 4]])
        #>>> kvar = K.variable(value=val)
        #>>> K.ndim(inputs)
        3
        #>>> K.ndim(kvar)
        2
    ```
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


# 我们要对序列做 Mask 以忽略填充部分的影响。一般的 Mask 是将填充部分置零，但 Attention 中的 Mask 是要在 softmax 之前，
# 把填充部分减去一个大整数（这样 softmax 之后就非常接近 0 了）
def mask_logits(inputs, mask, mask_value=-1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)