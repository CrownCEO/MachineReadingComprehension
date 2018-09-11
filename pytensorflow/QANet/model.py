import tensorflow as tf
from pytensorflow.QANet.layers import regularizer, conv, highway, residual_block, optimized_trilinear_for_attention, mask_logits, total_params


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

            # self.word_unk = tf.get_variable("word_unk", shape = [config.glove_dim], initializer=initializer())
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
                    word_mat, dtype=tf.float32), trainable=False)
            self.char_mat = tf.get_variable(
                    "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
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
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
                grads = self.opt.compute_gradients(self.loss)
                # zip(*) 为解压
                gradients, variables = zip(*grads)
                # Gradient Clipping的引入是为了处理gradient explosion或者gradients vanishing的问题
                capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                # 这里使用了optimizer.apply_gradients来将求得的梯度用于参数修正，而不是之前简单的optimizer.minimize(cost)
                self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        # PL 代表 para_length 和c_maxlen是一个东西
        # PL=400 QL=50 CL=16 d=96 dc=64
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads

        # variable_scope/name_scope简介这种scope最直接的影响是: 所有在scope下面创建的variable都会把这个scope的名字作为前缀
        # name_scope给operation分类。同时使用variable_scope来区分variable.

        # word embedding + character embedding，word embedding从预先训练好的词向量中读取，每个词向量维度为p1，假设词w对应词向量为xw；
        # character embedding随机初始化，维度为p2，在此基础上将每个词维度padding or truncating为k，则每个词可以表示成一个p2∗k的矩阵，
        # 经过卷积和max - pooling后得到一个p2维的character - level的词向量，记为xc。
        # 将xw​和xc​拼接，得到w对应词向量[xw;xc]∈R(p1 + p2)​，最后将拼接的词向量通过一个两层的highway network，其输出即为embedding层的输出。
        with tf.variable_scope("Input_Embedding_Layer"):
            # ch_emb 和 qh_emb 之所以 传进去 char_mat  是因为 ch和qh都是字母索引
            # char_mat : M * 64    ch: N * c_maxlen * 16
            # CL * dc 大小的矩阵 有 N * PL 个
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.qh), [N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            # Bidaf style conv-highway encoder
            # [N * 400, 12, 96]
            ch_emb = conv(ch_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=None)
            qh_emb = conv(qh_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=True)

            # [N * 400，1，96]
            ch_emb = tf.reduce_max(ch_emb, axis=1)
            qh_emb = tf.reduce_max(qh_emb, axis=1)

            # [N ,400, 96]
            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])

            # [N,400,300]
            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            # [N,400,396]
            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)
            # [N,400,96]
            c_emb = highway(c_emb, size=d, scope="highway", dropout=self.dropout, reuse=None)
            q_emb = highway(q_emb, size=d, scope="highway", dropout=self.dropout, reuse=True)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb, num_blocks=1, num_conv_layers=4, kernel_size=7, mask=self.c_mask, num_filters=d,
                               num_heads=nh,
                               seq_len=self.c_len, scope="Encoder_Residual_Block")
            q = residual_block(q_emb, num_blocks=1, num_conv_layers=4, kernel_size=7, mask=self.q_mask, num_filters=d,
                               num_heads=nh,
                               seq_len=self.q_len, scope="Encoder_Residual_Block", reuse=True,
                               # Share the weights between passage and question
                               bias=False, dropout=self.dropout)

        # 计算context-to-query attention和query-to-context attention矩阵。 1.
        # 分别用C和Q来表示编码后的context和query，首先计算context和query的相似性矩阵，记为S∈Rn×m，其中相似度计算公式为：f(q,c)=W0[q,c,
        # q⊙c]。 2. 用softmax对S的行、列分别做归一化得到S_、S_T，则 context-to-query attention矩阵A=S_⋅QT∈Rn×d，query-to-context
        # attention矩阵B=S_⋅S_T⋅CT
        # DCN 中介绍了CoAttention相关，而BIDAF中的Attention Flow Layer和 Context_to_Query_Attention_Layer 有点相似
        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen,
                                                  input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]

        # 与BIDAF类似，每个位置的输入是[c,a,c⊙a,c⊙b]，a和b分别为attention矩阵A和B的行。
        # 该层中encoder blocks的卷积层数为2，总blocks数为1，其余参数与embedding encoder layer相同。
        # 三个model encoder之间共享参数。
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis=-1)
            self.enc = [conv(inputs, d, name="input_projection")]
            for i in range(3):
                if i % 2 == 0:  # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                                   num_blocks=7,
                                   num_conv_layers=2,
                                   kernel_size=5,
                                   mask=self.c_mask,
                                   num_filters=d,
                                   num_heads=nh,
                                   seq_len=self.c_len,
                                   scope="Model_Encoder",
                                   bias=False,
                                   reuse=True if i > 0 else None,
                                   dropout=self.dropout)
                )
        # 分别预测每个位置是answer span的起始点和结束点的概率，分别记为p1、p2，计算公式如下：
        # p1=softmax(W1[M0;M1])；p2=softmax(W2[M0;M2])
        # 其中&W_1&、&W_2&是两个可训练变量，M0、M1、M2依次对应结构图中三个model encoder block的输出（自底向上）。
        # 目标函数：
        # L(θ)=−1N∑iN[log(p1y1i)+log(p2y2i)]
        # 其中y1i、y2i分别为第i个样本的groudtruth的起始位置和结束位置。
        # 对测试集进行预测时，span(s, e)的选取规则是：p1s、p2e最大且s≤e。
        with tf.variable_scope("Output_Layer"):
            start_logits = tf.squeeze(
                conv(tf.concat([self.enc[1], self.enc[2]], axis=-1), 1, bias=False, name="start_pointer"), -1)
            end_logits = tf.squeeze(
                conv(tf.concat([self.enc[1], self.enc[3]], axis=-1), 1, bias=False, name="end_pointer"), -1)
            self.logits = [mask_logits(start_logits, mask=self.c_mask),
                           mask_logits(end_logits, mask=self.c_mask)]

            logits1, logits2 = [l for l in self.logits]

            # 还没搞太懂
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            # input：张量。秩为 k 的张量。
            # num_lower：int64 类型的张量；0-D 张量；要保持的对角线的数量；如果为负，则保留整个下三角。
            # num_upper：int64 类型的张量；0-D 张量；要保留的 superdiagonals 数；如果为负，则保持整个上三角。
            outer = tf.matrix_band_part(outer, 0, config.ans_limit)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)

        if config.l2_norm is not None:
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        if config.decay is not None:
            # https://blog.csdn.net/PKU_Jade/article/details/73477112
            # 用来让模型不被更新的太快
            self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            # 构建的操作必须在ema_op之后执行才可以
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
