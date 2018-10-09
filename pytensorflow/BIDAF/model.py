import tensorflow as tf

from pytensorflow.BIDAF.layers import conv, highway, bidirectional_dynamic_rnn, optimized_trilinear_for_attention, \
    mask_logits, regularizer, total_params


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, demo=False, graph=None):
        self.config = config
        self.demo = demo
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            if self.demo:
                # None代表N行 config.test_para_limit代表1000列 即1000个单词,所以self.c是一个N行1000列的矩阵
                # self.c 和 self.q 存储的是单词的索引
                self.c = tf.placeholder(tf.int32, [None, config.test_para_limit], "context")
                self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit], "question")
                # N * 1000 * 16 N个段落，每个段落1000个单词，每个单词长度为16个字符   即 有N个1000 * 16 的向量
                self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit], "context_char")
                self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit], "question_char")
                self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index1")
                self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index2")
            else:
                # 分别是原文单词索引，问题单词索引，原文字母索引，问题字母索引，答案起始位置，答案结束位置，问题id
                self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=False)
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
            # 转换为bool 再转换为int32 是为了将 数据转换为 0，1序列
            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            # self.c_len 和 self.q_len 是N * 1 的
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

            if opt:
                # 32，16
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                # 计算0维度上元素最大值，在这里是返回batch 中原文最长的长度
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                # y1, y2 里面值是什么
                self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
                self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit
            # self.ch_len self.qh_len 是一个向量 [N * c_maxlen]
            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])
            self.forward()
            total_params()
            if trainable:
                self.lr = tf.minimum(config.learning_rate,
                                     0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        # PL 代表 para_length 和c_maxlen是一个东西
        # PL=400 QL=50 CL=16 d=96 dc=64
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads
        # 论文原文将模型分为了六层 Character Embedding Layer -> Word Embedding Layer -> Contextual Embedding Layer -> Attention Flow Layer -> Modeling Layer -> Output Layer
        # 在这里我们依然分为五层 Input embedding layer = Character Embedding Layer + Word Embedding Layer
        with tf.variable_scope("Input_Embedding_Layer"):
            self.ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc])
            self.qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc])
            self.ch_emb = conv(self.ch_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv",
                               reuse=None)
            self.qh_emb = conv(self.qh_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv",
                               reuse=True)

            # 使用了max_pool 其实这样做 忽略了 词语的位置信息
            # 现在有改进的措施是使用 K-Max pooling 或者 chunk-Max pool
            self.ch_emb = tf.reduce_max(self.ch_emb, axis=1)
            self.qh_emb = tf.reduce_max(self.qh_emb, axis=1)
            self.ch_emb = tf.reshape(self.ch_emb, [N, PL, self.ch_emb.shape[-1]])
            self.qh_emb = tf.reshape(self.qh_emb, [N, QL, self.ch_emb.shape[-1]])

            self.c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
            self.q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            self.c_emb = tf.concat([self.c_emb, self.ch_emb], axis=2)
            self.q_emb = tf.concat([self.q_emb, self.qh_emb], axis=2)

            self.c_emb = highway(self.c_emb, size=d, scope="highway", dropout=self.dropout, reuse=None)
            self.q_emb = highway(self.q_emb, size=d, scope="highway", dropout=self.dropout, reuse=True)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(d, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(d, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=0.5)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=0.5)
            (output_fw, output_bw), (output_fw_state, output_bw_state) = bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                                                   lstm_bw_cell,
                                                                                                   self.c_emb,
                                                                                                   dtype=tf.float32)
            self.H = tf.concat([output_fw, output_bw], 2)
            (output_fw, output_bw), (output_fw_state, output_bw_state) = bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                                                   lstm_bw_cell,
                                                                                                   self.q_emb,
                                                                                                   dtype=tf.float32)
            self.U = tf.concat([output_fw, output_bw], 2)
        with tf.variable_scope("Context_Query_Attention_Layer"):
            S = optimized_trilinear_for_attention([self.H, self.U], self.c_maxlen, self.q_maxlen,
                                                  input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, self.U)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), self.H)
            self.attention_outputs = [self.H, self.c2q, self.H * self.c2q, self.H * self.q2c]
        with tf.variable_scope("Model_Encoder_Layer"):
            self.G = tf.concat(self.attention_outputs, axis=-1)
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(d, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(d, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=0.5)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=0.5)
            (output_fw, output_bw), (output_fw_state, output_bw_state) = bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                                                   lstm_bw_cell, self.G,
                                                                                                   dtype=tf.float32)
            self.M = tf.concat([output_fw, output_bw], 2)
        with tf.variable_scope("Output_Layer"):
            self.M0 = self.G
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(d, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(d, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=0.5)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=0.5)
            (output_fw, output_bw), (output_fw_state, output_bw_state) = bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                                                   lstm_bw_cell, self.M,
                                                                                                   dtype=tf.float32)
            self.M2 = tf.concat([output_fw, output_bw], 2)
            self.p1 = tf.concat([self.M0, self.M], 2)
            self.p2 = tf.concat([self.M0, self.M2], 2)
            weights_p1 = tf.get_variable("weights_p1", [10 * d, 1])
            weights_p2 = tf.get_variable("weights_p2", [10 * d, 1])
            self.start_logits = tf.squeeze(conv(self.p1, 1, bias=False, name="start_pointer"), -1)
            self.end_logits = tf.squeeze(conv(self.p2, 1, bias=False, name="end_pointer"), -1)
            self.logits = [mask_logits(self.start_logits, mask=self.c_mask),
                           mask_logits(self.end_logits, mask=self.c_mask)]
            logits1, logits2 = [l for l in self.logits]
            self.outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                                   tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            self.outer = tf.matrix_band_part(self.outer, 0, config.ans_limit)
            self.yp1 = tf.argmax(tf.reduce_max(self.outer, axis=1), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(self.outer, axis=2), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)

        if config.l2_norm is not None:
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        if config.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)
                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var, v))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
