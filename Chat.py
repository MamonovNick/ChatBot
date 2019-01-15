import ModelSettings as Ms
import tensorlayer as tl
import tensorflow as tf
import random
import paraphrase as ph
import wikisearch as wiki


def generate_answer(question, word2idx, idx2word, unk_id, session, rnn_net, ec_seqs, dc_seqs, start_id, end_id, sm):
    seed_id = [word2idx.get(w, unk_id) for w in question.split(" ")]

    # Encode and get state
    state = session.run(rnn_net.final_state_encode,
                        {ec_seqs: [seed_id]})
    # Decode, feed start_id and get first word [https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py]
    o, state = session.run([sm, rnn_net.final_state_decode],
                           {rnn_net.initial_state_decode: state,
                            dc_seqs: [[start_id]]})
    w_id = tl.nlp.sample_top(o[0], top_k=3)
    w = idx2word[w_id]
    # Decode and feed state iteratively
    sentence = [w]
    for _ in range(30):  # max sentence length
        o, state = session.run([sm, rnn_net.final_state_decode],
                               {rnn_net.initial_state_decode: state,
                                dc_seqs: [[w_id]]})
        w_id = tl.nlp.sample_top(o[0], top_k=2)
        w = idx2word[w_id]
        if w_id == end_id:
            break
        sentence = sentence + [w]
    return sentence


def main():
    settings = Ms.BaseSettings()
    metadata, train_q, train_a, test_q, test_a, valid_q, valid_a = Ms.initial_setup(settings)
    src_len = len(train_q)
    tgt_len = len(train_a)
    assert src_len == tgt_len

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

    # Load Model
    tl.files.load_and_assign_npz(sess=session, name='model.npz', network=net)

    # TODO Replace simple first sentence to more complex
    if random.random() > 0.5:
        print(" >", "Hi")

    while True:
        question = input('You: ')
        if question.lower() == 'exit':
            break

        if settings.WikiSearch:
            result = wiki.wikipedia_search(question)
            if result:
                print(" >", ' '.join(result))
                continue

        sentence = generate_answer(question, word2idx, idx2word, unk_id, session, net_rnn,
                                   encode_seqs2, decode_seqs2, start_id, end_id, sm)

        # perform simple paraphraser
        if random.random() > settings.ParaphraseFrequency:
            sentence = ph.paraphrase(sentence)

        print(" >", ' '.join(sentence))

    session.close()


if __name__ == '__main__':
    main()
