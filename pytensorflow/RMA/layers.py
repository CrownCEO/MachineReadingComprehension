import tensorflow as tf
from tensor2tensor.layers.common_layers import conv1d
from tensorflow.contrib.keras import layers


def exp_mask(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def align_block(u, v, c_mask, q_mask, Lambda, filters=128, E_0=None, B_0=None, Z_0=None):
    with tf.variable_scope("Interactive_Alignment"):
        u_ = tf.nn.relu(conv1d(u, filters, 1, name="Wu"))
        v_ = tf.nn.relu(conv1d(v, filters, 1, name="Wv"))
        E = tf.matmul(v_, u_, transpose_b=True)  # [bs, len_q, len_c]
        if E_0 is not None:
            E += (Lambda * E_0)
        E_ = tf.nn.softmax(exp_mask(E, tf.expand_dims(q_mask, axis=-1)), axis=1)  # [bs, len_q, len_c]
        v_E = tf.matmul(E_, v, transpose_a=True)  # [bs, len_c, dim]

        # fusion
        uv = tf.concat([u, v_E, u * v_E, u - v_E], axis=-1)
        x_ = tf.nn.relu(conv1d(uv, filters, 1, name="Wr"))
        g = tf.nn.sigmoid(conv1d(uv, filters, 1, name="Wg"))
        o = g * x_ + (1 - g) * u  # [bs, len_c, dim]

    with tf.variable_scope("Self_Alignment"):
        # attention
        h_1 = tf.nn.relu(conv1d(o, filters, 1, name="Wh1"))
        h_2 = tf.nn.relu(conv1d(o, filters, 1, name="Wh2"))
        B = tf.matmul(h_2, h_1, transpose_b=True)  # [bs, len_c, len_c]
        if B_0 is not None:
            B += (Lambda * B_0)
        B_ = tf.nn.softmax(exp_mask(B, tf.expand_dims(c_mask, axis=-1)), axis=1)  # [bs, len_c, len_c]
        o_B = tf.matmul(B_, o, transpose_a=True)

        # fusion
        oo = tf.concat([o, o_B, o * o_B, o - o_B], axis=-1)
        x_ = tf.nn.relu(conv1d(oo, filters, 1, name="Wr"))
        g = tf.nn.sigmoid(conv1d(oo, filters, 1, name="Wg"))
        Z = g * x_ + (1 - g) * o  # [bs, len_c, dim]
    with tf.variable_scope("Evidence_Collection"):
        if Z_0 is not None:
            Z = tf.concat([Z, Z_0[0], Z_0[1]], axis=-1)
        R = layers.Bidirectional(layers.LSTM(filters // 2, return_sequences=True))(Z)  # [bs, len_c, dim]

    # return the E_t, B_t
    E_t = tf.nn.softmax(exp_mask(E, tf.expand_dims(c_mask, axis=1)), axis=-1)  # [bs, len_q, len_c]
    E_t = tf.matmul(E_t, B_)
    B_t = tf.nn.softmax(exp_mask(B, tf.expand_dims(c_mask, axis=1)), axis=-1)  # [bs, len_c, len_c]
    B_t = tf.matmul(B_t, B_)

    return R, Z, E_t, B_t


def start_logits(R, s, mask, filters=28):
    with tf.variable_scope("Start_Pointer"):
        logits1 = tf.concat([R, s, R * s, R - s], axis=-1)
        logits1 = tf.nn.tanh(conv1d(logits1, filters, 1, name="W1"))
        logits1 = tf.squeeze(conv1d(logits1, 1, 1, name="W1t"), axis=-1)
        logits1 = exp_mask(logits1, mask)
    return logits1


def end_logits(R, logits1, s, mask, filters=128):
    with tf.variable_scope("End_Pointer"):
        l = R * tf.expand_dims(logits1, axis=-1)  # [bs, len_c, dim]
        s_ = tf.concat([s, l, s * l, s - l], axis=-1)
        x = tf.nn.relu(conv1d(s_, filters, 1, name="Wr"))  # [bs, len_c, dim]
        g = tf.sigmoid(conv1d(s_, filters, 1, name="Wg"))  # [bs, len_c, dim]
        s_ = g * x + (1 - g) * s  # [bs, len_c, dim]

        logits2 = tf.concat([R, s_, R * s_, R - s_], axis=-1)
        logits2 = tf.nn.tanh(conv1d(logits2, filters, 1, name="W2"))
        logits2 = tf.squeeze(conv1d(logits2, 1, 1, name="W2t"), axis=-1)
        logits2 = exp_mask(logits2, mask)
    return logits2


def summary_vector(q_emb, c_maxlen, mask):
    with tf.variable_scope("Question_Summary"):
        alpha = tf.nn.softmax(exp_mask(tf.squeeze(conv1d(q_emb, 1, 1), axis=-1), mask))
        s = tf.expand_dims(alpha, axis=-1) * q_emb
        s = tf.reduce_sum(s, axis=1, keepdims=True)  # [bs, 1, dim]
        s = tf.tile(s, [1, c_maxlen, 1])  # [bs, len_c, dim]
    return s


def mask_to_start(score, start, score_mask_value=-1e30):
    score_mask = tf.cast(tf.ones_like(start) - tf.cumsum(start, axis=-1), tf.float32)
    return score + score_mask * score_mask_value


def get_tf_f1(y_pred, y_true):
    y_true = tf.cast(y_true, tf.float32)
    y_union = tf.clip_by_value(y_pred + y_true, 0, 1)  # [bs, c_maxlen]
    y_diff = tf.abs(y_pred - y_true)  # [bs, c_maxlen]
    num_same = tf.cast(tf.reduce_sum(y_union, axis=-1) - tf.reduce_sum(y_diff, axis=-1), tf.float32)  # [bs,]
    y_precision = num_same / (tf.cast(tf.reduce_sum(y_pred, axis=-1), tf.float32) + 1e-7)  # [bs,]
    y_recall = num_same / (tf.cast(tf.reduce_sum(y_true, axis=-1), tf.float32) + 1e-7)  # [bs,]
    y_f1 = (2.0 * y_precision * y_recall) / (tf.cast(y_precision + y_recall, tf.float32) + 1e-7)  # [bs,]
    return tf.clip_by_value(y_f1, 0, 1)


def rl_loss(logits_start, logits_end, y_start, y_end, c_maxlen, rl_loss_type, topk=None):
    assert rl_loss_type == 'DCRL' or rl_loss_type == 'SCST' or rl_loss_type == 'topk_DCRL'
    # get ground truth prediction
    # s:[0,1,0,0,0], e:[0,0,0,1,0]->[0,1,1,1,1]-[0,0,0,1,1]->[0,1,1,0,0]+e:[0,0,0,1,0]->pred:[0,1,1,1,0]
    y_start_cumsum = tf.cumsum(y_start, axis=-1)
    y_end_cumsum = tf.cumsum(y_end, axis=-1)
    # 之所以 做差是为了  取出 答案序列， 之所以加y_end 是因为 做差的时候 把y_end 给减去了
    ground_truth = y_start_cumsum - y_end_cumsum + y_end  # [bs, c_maxlen]

    # get greedy prediction
    greedy_start = tf.one_hot(tf.argmax(logits_start, axis=-1), c_maxlen,
                              axis=-1)  # [bs, c_maxlen]->[bs,]->[bs, c_maxlen]
    masked_logits_end = mask_to_start(logits_end, greedy_start)
    greedy_end = tf.one_hot(tf.argmax(masked_logits_end, axis=-1), c_maxlen, axis=-1)
    greedy_start_cumsum = tf.cumsum(greedy_start, axis=-1)
    greedy_end_cumsum = tf.cumsum(greedy_end, axis=-1)
    greedy_prediction = greedy_start_cumsum - greedy_end_cumsum + greedy_end  # [bs, c_maxlen]
    # get greedy f1
    greedy_f1 = get_tf_f1(greedy_prediction, ground_truth)

    # get sampled prediction (use tf.multinomial)
    sampled_start_ind = tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(logits_start)), 1),
                                   axis=-1)  # [bs, c_maxlen]->[bs, 1]->[bs,]
    sampled_start = tf.one_hot(sampled_start_ind, c_maxlen, axis=-1)  # [bs, c_maxlen]->[bs,]->[bs, c_maxlen]
    masked_logits_end = mask_to_start(logits_end, sampled_start)
    sampled_end_ind = tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(masked_logits_end)), 1), axis=-1)
    sampled_end = tf.one_hot(sampled_end_ind, c_maxlen, axis=-1)
    sampled_start_cumsum = tf.cumsum(sampled_start, axis=-1)
    sampled_end_cumsum = tf.cumsum(sampled_end, axis=-1)
    sampled_prediction = sampled_start_cumsum - sampled_end_cumsum + sampled_end  # [bs, c_maxlen]
    # get sampled f1
    sampled_f1 = get_tf_f1(sampled_prediction, ground_truth)

    reward = tf.stop_gradient(sampled_f1 - greedy_f1)  # (sampled - baseline)
    sampled_start_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_start, labels=sampled_start)
    sampled_end_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_end, labels=sampled_end)

    if rl_loss_type == 'DCRL':
        reward = tf.clip_by_value(reward, 0., 1e7)
        reward_greedy = tf.clip_by_value(tf.stop_gradient(greedy_f1 - sampled_f1), 0., 1e7)
        greedy_start_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_start, labels=greedy_start)
        greedy_end_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_end, labels=greedy_end)
        return tf.reduce_mean(reward * (sampled_start_loss + sampled_end_loss) + reward_greedy * (
                greedy_start_loss + greedy_end_loss)), sampled_f1, greedy_f1
    elif rl_loss_type == 'SCST':
        return tf.reduce_mean(reward * (sampled_start_loss + sampled_end_loss)), sampled_f1, greedy_f1


# 用于计算总参数个数
def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
