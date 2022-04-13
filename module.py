import tensorflow as tf
import numpy as np

def get_token_embeddings_word():
    
    emb_path = '../data/cail_thulac.npy'
    word_embedding = np.cast[np.float32](np.load(emb_path))
    

    with tf.variable_scope("shared_weight_matrix_word", reuse=tf.AUTO_REUSE):
        # embeddings = tf.constant(word_embedding, dtype=tf.float32, name="constant_embeddings")
        embeddings = tf.get_variable(name="word_embedding",
                                    dtype=tf.float32,
                                     shape=(164673, 200),
                                     initializer=tf.constant_initializer(word_embedding))

        embeddings = tf.concat((embeddings[:164672, :], tf.zeros(shape=[1, 200])), 0)
    return embeddings

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def dag_lstm(input):
    # input [batch_size, input_dim]
    input = tf.expand_dims(input, 1) # [N, 1, input_dim]
    hidden_size = 128
    with tf.variable_scope('dag', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('dag_lstm_task1', reuse=tf.AUTO_REUSE):
            y_task1, state_tuple_task1 = tf.compat.v1.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(hidden_size),
                input,
                dtype=tf.float32)
        # state_tuple_task: c, h
        # c: [N, dim]   h: [N, dim]
        # y_task [N, 1, output_dim]

        c_1_2 = tf.get_variable('c_1_2', [hidden_size, hidden_size], dtype=tf.float32)
        h_1_2 = tf.get_variable('h_1_2', [hidden_size, hidden_size], dtype=tf.float32)

        b_c_2 = tf.get_variable('b_c_2', [hidden_size], dtype=tf.float32)
        b_h_2 = tf.get_variable('b_h_2', [hidden_size], dtype=tf.float32)

        initial_c_1 = tf.matmul(state_tuple_task1[0], c_1_2) + b_c_2
        initial_h_1 = tf.matmul(state_tuple_task1[1], h_1_2) + b_h_2

        initial_tuple_2 = tf.nn.rnn_cell.LSTMStateTuple(initial_c_1, initial_h_1)

        with tf.variable_scope('dag_lstm_task2', reuse=tf.AUTO_REUSE):
            y_task2, state_tuple_task2 = tf.compat.v1.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(hidden_size),
                input,
                initial_state=initial_tuple_2,
                dtype=tf.float32)

        c_1_3 = tf.get_variable('c_1_3', [hidden_size, hidden_size], dtype=tf.float32)
        c_2_3 = tf.get_variable('c_2_3', [hidden_size, hidden_size], dtype=tf.float32)

        h_1_3 = tf.get_variable('h_1_3', [hidden_size, hidden_size], dtype=tf.float32)
        h_2_3 = tf.get_variable('h_2_3', [hidden_size, hidden_size], dtype=tf.float32)

        b_c_3 = tf.get_variable('b_c_3', [hidden_size], dtype=tf.float32)
        b_h_3 = tf.get_variable('b_h_3', [hidden_size], dtype=tf.float32)

        initial_c_2 = tf.matmul(state_tuple_task1[0], c_1_3) + tf.matmul(state_tuple_task2[0], c_2_3) + b_c_3
        initial_h_2 = tf.matmul(state_tuple_task1[1], h_1_3) + tf.matmul(state_tuple_task1[1], h_2_3) + b_h_3

        initial_tuple_3 = tf.nn.rnn_cell.LSTMStateTuple(initial_c_2, initial_h_2)

        with tf.variable_scope('dag_lstm_task3', reuse=tf.AUTO_REUSE):
            y_task3, state_tuple_task3 = tf.compat.v1.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(hidden_size),
                input,
                initial_state=initial_tuple_3,
                dtype=tf.float32)

    return state_tuple_task1[1], state_tuple_task2[1], state_tuple_task3[1]
