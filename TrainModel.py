import ModelSettings as Ms
import Chat
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
from sklearn.utils import shuffle


def train(settings):
    metadata, train_q, train_a, test_q, test_a, valid_q, valid_a = Ms.initial_setup(settings)
    src_len = len(train_q)
    tgt_len = len(train_a)
    assert src_len == tgt_len

    n_step = src_len // settings.BatchSize
    src_vocab_size = len(metadata['idx2w'])
    emb_dim = settings.EmbeddingSize

    word2idx = metadata['w2idx']  # dict  word 2 index
    idx2word = metadata['idx2w']  # list index 2 word

    unk_id = word2idx['unk']  # 1

    start_id = src_vocab_size
    end_id = src_vocab_size + 1

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']

    src_vocab_size = src_vocab_size + 2

    ''' Seq2Seq wait for data in format:
    input_seqs : ['how', 'are', 'you', '<PAD_ID'>]
    decode_seqs : ['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]
    target_seqs : ['I', 'am', 'fine', '<END_ID>', '<PAD_ID'>]
    target_mask : [1, 1, 1, 1, 0]
    '''

    # Preprocessing
    target_seqs = tl.prepro.sequences_add_end_id([train_a[10]], end_id=end_id)[0]
    decode_seqs = tl.prepro.sequences_add_start_id([train_a[10]], start_id=start_id, remove_last=False)[0]
    target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]
    print("encode_seqs", [idx2word[i] for i in train_q[10]])
    print("target_seqs", [idx2word[i] for i in target_seqs])
    print("decode_seqs", [idx2word[i] for i in decode_seqs])
    print("target_mask", target_mask)
    print(len(target_seqs), len(decode_seqs), len(target_mask))

    # Init Session
    tf.reset_default_graph()
    session = tf.Session(config=settings.SessionConfig)

    # Training Data Placeholders
    encode_seqs = tf.placeholder(dtype=tf.int64, shape=[settings.BatchSize, None], name="encode_seqs")
    decode_seqs = tf.placeholder(dtype=tf.int64, shape=[settings.BatchSize, None], name="decode_seqs")
    target_seqs = tf.placeholder(dtype=tf.int64, shape=[settings.BatchSize, None], name="target_seqs")
    target_mask = tf.placeholder(dtype=tf.int64, shape=[settings.BatchSize, None], name="target_mask")

    net_out, _ = Ms.create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim, is_train=True, reuse=False)
    net_out.print_params(False)

    # Inference Data Placeholders
    encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
    decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")

    net, net_rnn = Ms.create_model(encode_seqs2, decode_seqs2, src_vocab_size, emb_dim, is_train=False, reuse=True)
    sm = tf.nn.softmax(net.outputs)

    # Loss Function
    loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs,
                                               input_mask=target_mask, return_details=False, name='cost')

    # Optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=settings.LearningRate).minimize(loss)

    # Init Vars
    session.run(tf.global_variables_initializer())

    initials = ["Have a good day",
                "Discover interesting projects and people to populate your personal news feed"]
    for epoch in range(settings.EpochNum):
        train_q, train_a = shuffle(train_q, train_a, random_state=0)
        total_loss, n_iter = 0, 0
        for q, a in tqdm(
                tl.iterate.minibatches(inputs=train_q, targets=train_a, batch_size=settings.BatchSize, shuffle=False),
                total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, settings.EpochNum), leave=False):
            q = tl.prepro.pad_sequences(q)
            _target_seqs = tl.prepro.sequences_add_end_id(a, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs)
            _decode_seqs = tl.prepro.sequences_add_start_id(a, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)
            # Uncomment to view the data here
            # for i in range(len(q)):
            #     print(i, [idx2word[id] for id in q[i]])
            #     print(i, [idx2word[id] for id in a[i]])
            #     print(i, [idx2word[id] for id in _target_seqs[i]])
            #     print(i, [idx2word[id] for id in _decode_seqs[i]])
            #     print(i, _target_mask[i])
            #     print(len(_target_seqs[i]), len(_decode_seqs[i]), len(_target_mask[i]))
            _, loss_iter = session.run([train_op, loss], {encode_seqs: q, decode_seqs: _decode_seqs,
                                                          target_seqs: _target_seqs, target_mask: _target_mask})
            total_loss += loss_iter
            n_iter += 1

        # printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, settings.EpochNum, total_loss / n_iter))

        # inference after every epoch
        for initial in initials:
            print("Query >", initial)
            for _ in range(5):
                sentence = Chat.generate_answer(initial, word2idx, idx2word, unk_id, session, net_rnn,
                                                encode_seqs2, decode_seqs2, start_id, end_id, sm)
                print(" >", ' '.join(sentence))

        # saving the model
        tl.files.save_npz(net.all_params, name='model.npz', sess=session)

    # session cleanup
    session.close()


def main():
    settings = Ms.BaseSettings()
    train(settings)


if __name__ == '__main__':
    main()
