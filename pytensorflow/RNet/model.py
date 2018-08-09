import tensorflow as tf
from params import Params

from pytensorflow.RNet.GRU import SRUCell, GRUCell, gated_attention_Wrapper
from pytensorflow.RNet.layers import encoding, bidirectional_GRU, apply_dropout, attention_rnn


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
