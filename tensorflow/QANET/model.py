import tensorflow as tf
from .layers import conv, highway

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
                self.c = tf.placeholder(tf.int32, [None, config.test_para_limit], "context")
                self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit], "question")
                self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit], "context_char")
                self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit], "question_char")
                self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index1")
                self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index2")
            else:
                self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()

            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=False)
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

            # 应该是先计算完长度 再转换为bool吧
            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

            if opt:
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
                self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            else:
                self.c_maxlen, self.q_maxlen, = config.para_limit, config.test_ques_limit

            self.ch_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

            self.forward()
            total_params()

            if trainable:
                self.lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.) *
                                     tf.log(tf.cast(self.global_step, tf.float32) + 1))
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
        # 代码是不是有问题 c_maxlen 应该是对应CL吧
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads

        # variable_scope/name_scope简介这种scope最直接的影响是: 所有在scope下面创建的variable都会把这个scope的名字作为前缀
        # name_scope给operation分类。同时使用variable_scope来区分variable.

        # word embedding + character embedding，word embedding从预先训练好的词向量中读取，每个词向量维度为p1，假设词w对应词向量为xw；
        # character embedding随机初始化，维度为p2，在此基础上将每个词维度padding or truncating为k，则每个词可以表示成一个p2∗k的矩阵，
        # 经过卷积和max - pooling后得到一个p2维的character - level的词向量，记为xc。
        # 将xw​和xc​拼接，得到w对应词向量[xw;xc]∈R(p1 + p2)​，最后将拼接的词向量通过一个两层的highway network，其输出即为embedding层的输出。
        with tf.variable_scope("Input_Embedding_layer"):
            ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            # Bidaf style conv-highway encoder
            ch_emb = conv(ch_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=None)
            qh_emb = conv(qh_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=True)

            ch_emb = tf.reduce_max(ch_emb, axis=1)
            qh_emb = tf.reduce_max(qh_emb, axis=1)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, size=d, scope="highway", dropout=self.dropout, reuse=None)
            q_emb = highway(q_emb, size=d, scope="highway", dropout=self.dropout, reuse=True)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb, num_blocks=1, num_conv_layers=4, kernel_size=7, mask=self.c_mask, num_filters=d, num_heads=nh,
                               seq_len=self.c_len, scope="Encoder_Residual_Block")
            q = residual_block(q_emb, num_blocks=1, num_conv_layers=4, kernel_size=7, mask=self.q_mask, num_filters=d, num_heads=nh,
                               seq_len=self.q_len, scope="Encoder_Residual_Block", reuse=True,# Share the weights between passage and question
                               bias=False, dropout=self.dropout)




