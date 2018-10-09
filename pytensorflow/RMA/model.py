import tensorflow as tf
from tensorflow.contrib.keras import layers

from pytensorflow.RMA.layers import align_block, summary_vector, start_logits, end_logits, rl_loss, total_params


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, demo=False, graph=None,
                 elmo_path=None):
        self.config = config
        self.demo = demo
        self.elmo_path = elmo_path
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            self.filters = config.filters
            self.init_lambda = config.init_lambda
            self.rl_loss_type = config.rl_loss_type
            self.l2_norm = config.l2_norm
            self.ans_limit = config.ans_limit
            self.decay = config.decay
            self.learning_rate = config.learning_rate

            if self.demo:
                # None代表N行 config.test_para_limit代表1000列 即1000个单词,所以self.c是一个N行1000列的矩阵
                # self.c 和 self.q 存储的是单词的索引
                self.c = tf.placeholder(tf.int32, [None, config.test_para_limit], "context")
                self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit], "question")
                # N * 1000 * 16 N个段落，每个段落1000个单词，每个单词长度为16个字符   即 有N个1000 * 16 的向量
                self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit], "context_char")
                self.qh = tf.placeholder(tf.inr32, [None, config.test_ques_limit, config.char_limit], "question_char")
                self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index1")
                self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index2")
            else:
                self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=False)
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.ch_mask = tf.cast(self.ch, tf.bool)
            self.qh_mask = tf.cast(self.qh, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(self.ch_mask, tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(self.qh_mask, tf.int32), axis=2), [-1])

            if opt:
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
                self.q = tf.slice(self.c, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
                self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit
            if self.elmo_path is not None:
                print("elmo need to be added")
            self.forward()
            total_params()
            if trainable:
                self.lr = tf.minimum(config.learning_rate,
                                     0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        # PL=400 QL=50 CL=16 d=96 dc=64
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads
        with tf.variable_scope("Input_Embedding_Layer"):
            self.ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc])
            self.qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc])
            self.ch_mask = tf.reshape(self.ch_mask, [-1, CL])
            self.qh_mask = tf.reshape(self.qh_mask, [-1, CL])
            char_bilstm = layers.Bidirectional(layers.LSTM(d, name="char_bilstm"))
            self.ch_emb = char_bilstm(self.ch_emb)
            self.qh_emb = char_bilstm(self.qh_emb)
            self.ch_emb = tf.reshape(self.ch_emb, [-1, self.c_maxlen, self.ch_emb.shape[-1]])
            self.qh_emb = tf.reshape(self.qh_emb, [-1, self.q_maxlen, self.qh_emb.shape[-1]])

            self.c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
            self.q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            self.c_emb = tf.concat([self.c_emb, self.ch_emb], axis=-1)
            self.q_emb = tf.concat([self.q_emb, self.qh_emb], axis=-1)

            with tf.variable_scope("BiLSTM_Embedding_Layer"):
                inputs_bilstm = layers.Bidirectional(
                    layers.LSTM(self.filters // 2, name="inputs_bilstm", return_sequences=True))
                self.c_emb = tf.nn.dropout(inputs_bilstm(self.c_emb, mask=self.c_mask), 1.0 - self.dropout)
                self.q_emb = tf.nn.dropout(inputs_bilstm(self.q_emb, mask=self.q_mask), 1.0 - self.dropout)

        with tf.variable_scope("Iterative_Aligner_Layer"):
            self.Lambda = tf.get_variable('Lambda', dtype=tf.float32, initializer=self.init_lambda)
            with tf.variable_scope("Single_Aligning_Block_1"):
                R1, Z1, E1, B1 = align_block(u=self.c_emb, v=self.q_emb,
                                             c_mask=self.c_mask, q_mask=self.q_mask,
                                             Lambda=self.Lambda, filters=self.filters)
                R1 = tf.nn.dropout(R1, 1.0 - self.dropout)
            with tf.variable_scope("Single_Aligning_Block_2"):
                R2, Z2, E2, B2 = align_block(u=R1, v=self.q_emb,
                                             c_mask=self.c_mask, q_mask=self.q_mask,
                                             E_0=E1, B_0=B1,
                                             Lambda=self.Lambda, filters=self.filters)
                R2 = tf.nn.dropout(R2, 1.0 - self.dropout)
            with tf.variable_scope("Single_Aligning_Block_3"):
                R3, Z3, E3, B3 = align_block(u=R2, v=self.q_emb,
                                             c_mask=self.c_mask, q_mask=self.q_mask,
                                             E_0=E2, B_0=B2, Z_0=[Z1, Z2],
                                             Lambda=self.Lambda, filters=self.filters)
                R3 = tf.nn.dropout(R3, 1.0 - self.dropout)
            self.R = R3

        with tf.variable_scope("Answer_Pointer"):
            s = summary_vector(self.q_emb, self.c_maxlen, mask=self.q_mask)
            logits1 = start_logits(self.R, s, mask=self.c_mask, filters=self.filters)
            logits2 = end_logits(self.R, logits1, s, mask=self.c_mask, filters=self.filters)

        start_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=self.y1)
        end_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=self.y2)
        self.loss = tf.reduce_mean(start_loss + end_loss)
        with tf.variable_scope("Reinforcement_Loss"):
            self.theta_a = tf.get_variable(name="theta_a", dtype=tf.float32, initializer=1.0)
            self.theta_b = tf.get_variable(name="theta_b", dtype=tf.float32, initializer=1.0)
            if self.rl_loss_type is not None:
                self.rl_loss, self.sampled_f1, self.greedy_f1 = rl_loss(logits1, logits2, self.y1, self.y2,
                                                                        self.c_maxlen, self.rl_loss_type)
                self.loss = (1 / (2 * (self.theta_a ** 2) + 1e-7)) * self.loss + (
                        1 / (2 * (self.theta_b ** 2) + 1e-7)) * self.rl_loss + tf.log(self.theta_a ** 2) + tf.log(
                    self.theta_b ** 2)
        # l2 loss
        if self.l2_norm is not None:
            decay_costs = []
            for var in tf.trainable_variables():
                decay_costs.append(tf.nn.l2_loss(var))
            self.loss += tf.multiply(self.l2_norm, tf.add_n(decay_costs))

        # output
        outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                          tf.expand_dims(tf.nn.softmax(logits2), axis=1))
        outer = tf.matrix_band_part(outer, 0, self.ans_limit)
        self.output1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1, name='start_output')
        self.output2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1, name='end_output')
        self.temp = self.output1
        self.temp1 = self.output2
        self.temp2 = self.temp1

        # EMA
        if self.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(self.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)
                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v is not None:
                        self.assign_vars.append(tf.assign(var, v))
