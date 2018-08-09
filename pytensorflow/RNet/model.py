import tensorflow as tf
from params import Params

from pytensorflow.RNet.GRU import SRUCell, GRUCell, gated_attention_Wrapper
from pytensorflow.RNet.layers import get_attn_params, encoding, bidirectional_GRU, apply_dropout, attention_rnn, \
    pointer_net, cross_entropy, total_params

optimizer_factory = {"adadelta": tf.train.AdadeltaOptimizer,
                     "adam": tf.train.AdamOptimizer,
                     "gradientdescent": tf.train.GradientDescentOptimizer,
                     "adagrad": tf.train.AdagradOptimizer}


class Model(object):
    def __init__(self, is_training=True, graph=None, demo=False):
        self.is_training = is_training
        self.demo = demo
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            if self.demo:
                self.passage_w = tf.placeholder(tf.int32, [1, Params.max_p_len, ], "passage_w")
                self.question_w = tf.placeholder(tf.int32, [1, Params.max_q_len, ], "passage_q")
                self.passage_c = tf.placeholder(tf.int32, [1, Params.max_p_len, Params.max_char_len], "passage_pc")
                self.question_c = tf.placeholder(tf.int32, [1, Params.max_q_len, Params.max_char_len], "passage_qc")
                self.passage_w_len_ = tf.placeholder(tf.int32, [1, 1], "passage_w_len_")
                self.question_w_len_ = tf.placeholder(tf.int32, [1, 1], "question_w_len")
                self.passage_c_len = tf.placeholder(tf.int32, [1, 1], "question_w_len_")
                self.question_c_len = tf.placeholder(tf.int32, [1, Params.max_q_len], "question_c_len")
                self.data = (self.passage_w, self.question_w, self.passage_c, self.question_c, self.passage_w_len_,
                             self.question_w_len_,
                             self.passage_c_len, self.question_c_len)
            else:
                self.data, self.num_batch = get_batch(is_training=is_training)
                (self.passage_w,
                 self.question_w,
                 self.passage_c,
                 self.question_c,
                 self.passage_w_len_,
                 self.question_w_len_,
                 self.passage_c_len,
                 self.question_c_len,
                 self.indices) = self.data

            self.passage_w_len = tf.squeeze(self.passage_w_len_, -1)
            self.question_w_len = tf.squeeze(self.question_w_len_, -1)
            self.forward()

            if is_training:
                self.loss_function()
                self.summary()
                self.init_op = tf.gloabal_variables_initialzer()
            total_params()

    def forward(self):
        with tf.device('/cpu:0'):
            self.char_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.char_vocab_size, Params.char_emb_size]),
                                               trainable=True, name="char_embeddings")
            self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.vocab_size, Params.emb_size]),
                                               trainable=False, name="word_embeddings")
            self.word_embeddings_placeholder = tf.placeholder(tf.float32, [Params.vocab_size, Params.emb_size],
                                                              "word_embeddings_placeholder")
            self.emb_assign = tf.assign(self.word_embeddings, self.word_embeddings_placeholder)

            # Embed the question and passage information for word and character tokens
        self.passage_word_encoded, self.passage_char_encoded = encoding(self.passage_w,
                                                                        self.passage_c,
                                                                        word_embeddings=self.word_embeddings,
                                                                        char_embeddings=self.char_embeddings,
                                                                        scope="passage_embeddings")
        self.question_word_encoded, self.question_char_encoded = encoding(self.question_w,
                                                                          self.question_c,
                                                                          word_embeddings=self.word_embeddings,
                                                                          char_embeddings=self.char_embeddings,
                                                                          scope="question_embeddings")

        self.passage_char_encoded = bidirectional_GRU(self.passage_char_encoded,
                                                      self.passage_c_len,
                                                      cell_fn=SRUCell if Params.SRU else GRUCell,
                                                      scope="passage_char_encoding",
                                                      output=1,
                                                      is_training=self.is_training)
        self.question_char_encoded = bidirectional_GRU(self.question_char_encoded,
                                                       self.question_c_len,
                                                       cell_fn=SRUCell if Params.SRU else GRUCell,
                                                       scope="question_char_encoding",
                                                       output=1,
                                                       is_training=self.is_training)
        self.passage_encoding = tf.concat((self.passage_word_encoded, self.passage_char_encoded), axis=2)
        self.question_encoding = tf.concat((self.question_word_encoded, self.question_char_encoded), axis=2)

        # Passage and question encoding
        # cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _ in range(2)]
        self.passage_encoding = bidirectional_GRU(self.passage_encoding,
                                                  self.passage_w_len,
                                                  cell_fn=SRUCell if Params.SRU else GRUCell,
                                                  layers=Params.num_layers,
                                                  scope="passage_encoding",
                                                  output=0,
                                                  is_training=self.is_training)
        # cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _
        # in range(2)]
        self.question_encoding = bidirectional_GRU(self.question_encoding,
                                                   self.question_w_len,
                                                   cell_fn=SRUCell if Params.SRU else GRUCell,
                                                   layers=Params.num_layers,
                                                   scope="question_encoding",
                                                   output=0,
                                                   is_training=self.is_training)

        self.params = get_attn_params(Params.attn_size, initializer=tf.contrib.layers.xavier_initializer)
        # Apply gated attention recurrent network for both query-passage matching and self matching networks
        with tf.variable_scope("gated_attention_based_recurrent_networks"):
            memory = self.question_encoding
            inputs = self.passage_encoding
            scopes = ["question_passage_matching", "self_matching"]
            params = [(([self.params["W_u_Q"],
                         self.params["W_u_P"],
                         self.params["W_v_P"]], self.params["v"]),
                       self.params["W_g"]),
                      (([self.params["W_v_P_2"],
                         self.params["W_v_Phat"]], self.params["v"]),
                       self.params["W_g"])]
            for i in range(2):
                args = {"num_units": Params.attn_size,
                        "memory": memory,
                        "params": params[i],
                        "self_matching": False if i == 0 else True,
                        "memory_len": self.question_w_len if i == 0 else self.passage_w_len,
                        "is_training": self.is_training,
                        "use_SRU": Params.SRU}
                cell = [
                    apply_dropout(gated_attention_Wrapper(**args), size=inputs.shape[-1], is_training=self.is_training)
                    for _ in range(2)]
                inputs = attention_rnn(inputs,
                                       self.passage_w_len,
                                       Params.attn_size,
                                       cell,
                                       scope=scopes[i])
                memory = inputs  # self matching (attention over itself)
            self.self_matching_output = inputs
        self.final_bidirectional_outputs = bidirectional_GRU(self.self_matching_output, self.passage_w_len,
                                                             cell_fn=SRUCell if Params.SRU else GRUCell,
                                                             scope="bidirectional_readout",
                                                             output=0, is_training=self.is_training)

        # pointer networks
        params = (([self.params["W_u_Q"], self.params["W_v_Q"]], self.params["v"]),
                  ([self.params["W_h_P"], self.params["W_h_a"]], self.params["v"]))
        cell = apply_dropout(GRUCell(Params.attn_size * 2), size=self.final_bidirectional_outputs.shape[-1],
                             is_training=self.is_training)
        self.points_logits = pointer_net(self.final_bidirectional_outputs, self.passage_w_len,
                                         self.question_encoding, self.question_w_len, cell, params,
                                         scope="pointer_network")

        self.logit_1, self.logit_2 = tf.split(self.points_logits, 2, axis=1)
        self.logit_1 = tf.transpose(self.logit_1, [0, 2, 1])
        self.dp = tf.matmul(self.logit_1, self.logit_2)
        self.dp = tf.matrix_band_part(self.dp, 0, 15)
        self.output_index_1 = tf.argmax(tf.reduce_max(self.dp, axis=2), -1)
        self.output_index_2 = tf.argmax(tf.reduce_max(self.dp, axis=1), -1)
        self.output_index = tf.stack([self.output_index_1, self.output_index_2], axis=1)
        # self.output_index = tf.argmax(self.points_logits, axis = 2)

    def loss_function(self):
        with tf.variable_scope("loss"):
            shapes = self.passage_w.shape
            self.indices_prob = tf.one_hot(self.indices, shapes[1])
            self.mean_loss = cross_entropy(self.points_logits, self.indices_prob)
            self.optimizer = optimizer_factory[Params.optimizer](**Params.opt_arg[Params.optimizer])

            if Params.clip:
                # gradient clipping by norm
                gradients, variables = zip(*self.optimizer.compute_gradients(self.mean_loss))
                gradients, _ = tf.clip_by_global_norm(gradients, Params.norm)
                self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            else:
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

    def summary(self):
        self.F1 = tf.Variable(tf.constant(0.0, shape=(), dtype=tf.float32), trainable=False, name="F1")
        self.F1_placeholder = tf.placeholder(tf.float32, shape=(), name="F1_placeholder")
        self.EM = tf.Variable(tf.constant(0.0, shape=(), dtype=tf.float32), trainable=False, name="EM")
        self.EM_placeholder = tf.placeholder(tf.float32, shape=(), name="EM_placeholder")
        self.dev_loss = tf.Variable(tf.constant(5.0, shape=(), dtype=tf.float32), trainable=False, name="dev_loss")
        self.dev_loss_placeholder = tf.placeholder(tf.float32, shape=(), name="dev_loss")
        self.metric_assign = tf.group(tf.assign(self.F1, self.F1_placeholder), tf.assign(self.EM, self.EM_placeholder),
                                      tf.assign(self.dev_loss, self.dev_loss_placeholder))
        tf.summary.scalar('loss_training', self.mean_loss)
        tf.summary.scalar('loss_dev', self.dev_loss)
        tf.summary.scalar("F1_Score", self.F1)
        tf.summary.scalar("Exact_Match", self.EM)
        tf.summary.scalar('learning_rate', Params.opt_arg[Params.optimizer]['learning_rate'])
        self.merged = tf.summary.merge_all()
