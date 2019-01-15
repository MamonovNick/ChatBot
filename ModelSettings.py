
from data.TwitterCorpus import data
import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import DenseLayer, EmbeddingInputlayer, Seq2Seq, retrieve_seq_length_op2


class BaseSettings:
    def __init__(self):
        self.DataCorpus = "MovieDialog"
        self.BatchSize = 32
        self.EpochNum = 50
        self.LearningRate = 0.001
        self.EmbeddingSize = 1024
        self.TrainPart = 0.7
        self.TestPart = 0.15
        self.ValidPart = 0.15
        self.SessionConfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.ParaphraseFrequency = 2
        self.WikiSearch = 1


def initial_setup(settings):
    metadata, idx_q, idx_a = data.load_data(PATH='data/{}Corpus/'.format(settings.DataCorpus))
    (train_q, train_a), (test_q, test_a), (valid_q, valid_a) = data.split_dataset(idx_q, idx_a)
    train_q = tl.prepro.remove_pad_sequences(train_q.tolist())
    train_a = tl.prepro.remove_pad_sequences(train_a.tolist())
    test_q = tl.prepro.remove_pad_sequences(test_q.tolist())
    test_a = tl.prepro.remove_pad_sequences(test_a.tolist())
    valid_q = tl.prepro.remove_pad_sequences(valid_q.tolist())
    valid_a = tl.prepro.remove_pad_sequences(valid_a.tolist())
    return metadata, train_q, train_a, test_q, test_a, valid_q, valid_a


def create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim, is_train=True, reuse=False):
    # Create the LSTM model
    with tf.variable_scope("model", reuse=reuse):
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs=encode_seqs,
                vocabulary_size=src_vocab_size,
                embedding_size=emb_dim,
                name='seq_embedding')
            vs.reuse_variables()
            net_decode = EmbeddingInputlayer(
                inputs=decode_seqs,
                vocabulary_size=src_vocab_size,
                embedding_size=emb_dim,
                name='seq_embedding')

        net_rnn = Seq2Seq(net_encode, net_decode,
                          cell_fn=tf.nn.rnn_cell.LSTMCell,
                          n_hidden=emb_dim,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1),
                          encode_sequence_length=retrieve_seq_length_op2(encode_seqs),
                          decode_sequence_length=retrieve_seq_length_op2(decode_seqs),
                          initial_state_encode=None,
                          dropout=(0.5 if is_train else None),
                          n_layer=3,
                          return_seq_2d=True,
                          name='seq2seq')

        net_out = DenseLayer(net_rnn, n_units=src_vocab_size, act=tf.identity, name='output')
    return net_out, net_rnn
