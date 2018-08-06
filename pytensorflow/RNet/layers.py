import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell
from params import Params
from zoneout import ZoneoutWrapper


def encoding(word, char, word_embeddings, char_embeddings, scope="embedding"):
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
        char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
        return word_encoding, char_encoding


def bidirectional_GRU(inputs, inputs_len, cell=None, cell_fn=tf.contrib.rnn.GRUCell, units=Params.attn_size,
                      layers=1, scope="Bidirectional_GRU", output=0, is_training=True, reuse=None):
    """
    Bidirectional recurrent neural network with GRU cells.
    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    """
    with tf.variable_scope(scope, reuse=reuse):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            shapes = inputs.get_shape().as_list()
            if len(shapes) > 3:
                inputs = tf.reshape(inputs, (shapes[0] * shapes[1], shapes[2], -1))
                inputs_len = tf.reshape(inputs_len, (shapes[0] * shapes[1],))
            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(cell_fn(units), size=inputs.shape[-1] if i == 0 else units,
                                                      is_training=is_training) for i in range(layers)])
                cell_bw = MultiRNNCell([apply_dropout(cell_fn(units), size=inputs.shape[-1] if i == 0 else units,
                                                      is_training=is_training) for i in range(layers)])
            else:
                cell_fw, cell_bw = [apply_dropout(cell_fn(units), size=inputs.shape[-1], is_training=is_training)
                                    for _ in range(2)]
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=inputs_len,
                                                          dtype=tf.float32)
        if output == 0:
            return tf.concat(outputs, 2)
        elif output == 1:
            return tf.reshape(tf.concat(states, 1), (Params.batch_size, shapes[1], 2 * units))


def apply_dropout(inputs, size=None, is_training=True):
    """
    Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
    """
    if Params.dropout is None and Params.zoneout is None:
        return inputs
    if Params.zoneout is not None:
        return ZoneoutWrapper(inputs, state_zone_prob=Params.zoneout, is_training=is_training)
    elif is_training:
        return tf.contrib.rnn.DropoutWrapper(inputs, output_keep_prob=1 - Params.dropout,
                                             dtype=tf.float32)
    else:
        return inputs


def gated_attention(memory, inputs, states, units, params, self_matching=False, memory_len=None,
                    scope="gated_attention"):
    with tf.variable_scope(scope):
        weights, W_g = params
        inputs_ = [memory, inputs]
        states = tf.reshape(states, (Params.batch_size, Params.attn_size))
        if not self_matching:
            inputs_.append(states)
        scores = attention(inputs_, units, weights, memory_len=memory_len)
        scores = tf.expand_dims(scores, -1)
        attention_pool = tf.reduce_sum(scores * memory, 1)
        inputs = tf.concat((inputs, attention_pool), axis=1)
        g_t = tf.sigmoid(tf.matmul(inputs, W_g))
        return g_t * inputs
