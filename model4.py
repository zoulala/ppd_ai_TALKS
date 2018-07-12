from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os



class Model(object):
    '''seg_word ---> bilstm --> state.h-->  h*w+b --> h*w+b --> sigmoid
    result:  0.467(word) , ***(char)
    '''

    def __init__(self, num_classes, config, test=False, embeddings=None):
        self.num_classes = num_classes
        self.config = config
        self.embeddings = embeddings

        tf.reset_default_graph()
        self.build_nn()
        # if test is False:
        self.build_loss_optimizer()

        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def bilstm(self, seq, seq_len):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seq, sequence_length=seq_len,
                                                                         dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        return output, state

    def lstm(self, seq, seq_len):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        output, state = tf.nn.dynamic_rnn(cell_fw, seq, sequence_length=seq_len, dtype=tf.float32)
        return output, state

    def decode(self, initial_state):
        cell_fw = tf.nn.rnn_cell.LSTMCell(2 * self.config.hidden_size)
        dec_inputs = tf.zeros([self.config.batch_size, self.config.num_steps, 1])
        output, state = tf.nn.dynamic_rnn(cell_fw, dec_inputs, initial_state=initial_state)

        return output, state

    def activation(self, x):
        assert self.config.fc_activation in ["sigmoid", "relu", "tanh"]
        if self.config.fc_activation == "sigmoid":
            return tf.nn.sigmoid(x)
        elif self.config.fc_activation == "relu":
            return tf.nn.relu(x)
        elif self.config.fc_activation == "tanh":
            return tf.nn.tanh(x)

    def build_nn(self):
        ### Placeholders

        self.q1 = tf.placeholder( tf.int64, shape=[None, self.config.num_steps], name="question1")
        self.l1 = tf.placeholder( tf.int64, shape=[None],  name="len1")

        self.q2 = tf.placeholder( tf.int64, shape=[None, self.config.num_steps], name="question2")
        self.l2 = tf.placeholder( tf.int64, shape=[None], name="len2")

        self.y = tf.placeholder( tf.float32, shape=[None], name="is_duplicate")

        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(1, trainable=False,dtype=tf.float32, name="global_loss")

        ### Embedding
        if self.config.use_embedding is False:
            we1 = tf.one_hot(self.q1, depth=self.num_classes)  # 独热编码[1,2,3] depth=5 --> [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]，此时的输入节点个数为num_classes
            we2 = tf.one_hot(self.q2, depth=self.num_classes)
        else:
            with tf.device("/cpu:0"):
                embedding = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="embedding")
                # embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                we1 = tf.nn.embedding_lookup(embedding, self.q1)  # 词嵌入[1,2,3] --> [[3,...,4],[0.7,...,-3],[6,...,9]],embeding[depth*embedding_size]=[[0.2,...,6],[3,...,4],[0.7,...,-3],[6,...,9],[8,...,-0.7]]，此时的输入节点个数为embedding_size
                we2 = tf.nn.embedding_lookup(embedding, self.q2)
                we1 = tf.nn.dropout(we1, keep_prob=self.keep_prob)
                we2 = tf.nn.dropout(we2, keep_prob=self.keep_prob)

        ### ENCODER
        ### Shared layer
        with tf.variable_scope("bilstm") as scope:
            lstm1, state1 = self.bilstm(we1, self.l1)
            scope.reuse_variables()
            lstm2, state2 = self.bilstm(we2, self.l2)
            scope.reuse_variables()

        def mask_fn(x):
            return tf.sign(tf.reduce_sum(x, -1))

        def attn_pool(x, proj, alpha, masks):
            x = proj(x)
            align = tf.reduce_sum(alpha * tf.tanh(x), axis=-1)
            # masking
            paddings = tf.fill(tf.shape(align), float('-inf'))
            align = tf.where(tf.equal(masks, 0), paddings, align)
            # probability
            align = tf.expand_dims(tf.nn.softmax(align), -1)
            # weighted sum
            x = tf.squeeze(tf.matmul(x, align, transpose_a=True), -1)
            return x

        def query_context_attn(query, context, v, w_k, w_v, masks):
            query = tf.expand_dims(query, 1)
            keys = w_k(context)
            values = w_v(context)

            align = v * tf.tanh(query + keys)
            align = tf.reduce_sum(align, 2)

            paddings = tf.fill(tf.shape(align), float('-inf'))
            align = tf.where(tf.equal(masks, 0), paddings, align)

            align = tf.nn.softmax(align)
            align = tf.expand_dims(align, -1)
            val = tf.squeeze(tf.matmul(values, align, transpose_a=True), -1)
            return val

        mask1 = mask_fn(we1)
        mask2 = mask_fn(we2)
        with tf.variable_scope('attention_pooling'):
            proj = tf.layers.Dense(self.config.hidden_size)
            alpha = tf.get_variable('alpha', [self.config.hidden_size])
            attn_pool_1 = attn_pool(lstm1, proj, alpha, mask1)
            attn_pool_2 = attn_pool(lstm2, proj, alpha, mask2)

        with tf.variable_scope('query_context_attention'):
            v = tf.get_variable('v', [self.config.hidden_size])
            proj_k = tf.layers.Dense(self.config.hidden_size)
            proj_v = tf.layers.Dense(self.config.hidden_size)
            query_context_attn_1 = query_context_attn(attn_pool_1, lstm2, v, proj_k, proj_v, mask2)
            query_context_attn_2 = query_context_attn(attn_pool_2, lstm1, v, proj_k, proj_v, mask1)

        with tf.variable_scope('aggregation'):
            feat1 = attn_pool_1
            feat2 = attn_pool_2
            feat3 = tf.abs(feat1 - feat2)
            feat4 = feat1 * feat2
            feat5 = query_context_attn_1
            feat6 = query_context_attn_2
            feat7 = tf.abs(query_context_attn_1 - query_context_attn_2)
            feat8 = query_context_attn_1 * query_context_attn_2
            m1 = tf.reduce_max(we1, 1)
            m2 = tf.reduce_max(we2, 1)
            feat9 = tf.abs(m1 - m2)
            feat10 = m1 * m2

            x = tf.concat([feat1,
                           feat2,
                           feat3,
                           feat4,
                           feat5,
                           feat6,
                           feat7,
                           feat8,
                           feat9,
                           feat10], -1)
            x = tf.layers.dropout(x, 0.5, training=True)
            x = tf.layers.dense(x, 100, tf.nn.elu)
            x = tf.layers.dropout(x, 0.2, training=True)
            x = tf.layers.dense(x, 20, tf.nn.elu)

            self.x = tf.squeeze(tf.layers.dense(x, 1), -1)
        ### Features
        state1_fw = state1[0]
        state1_bw = state1[1]
        state1_h_concat = tf.concat(values=[state1_fw.h, state1_bw.h], axis=1)

        state2_fw = state2[0]
        state2_bw = state2[1]
        state2_h_concat = tf.concat(values=[state2_fw.h, state2_bw.h], axis=1)

        flat1 = state1_h_concat
        flat2 = state2_h_concat
        mult = tf.multiply(flat1, flat2)
        diff = tf.abs(tf.subtract(flat1, flat2))

        if self.config.feats == "raw":
            concat = tf.concat([flat1, flat2], axis=-1)
        elif self.config.feats == "dist":
            concat = tf.concat([mult, diff], axis=-1)
        elif self.config.feats == "all":
            concat = tf.concat([flat1, flat2, mult, diff], axis=-1)

        ### FC layers
        self.concat_size = int(concat.get_shape()[1])
        intermediary_size = 2 + (self.concat_size - 2) // 2
        # intermediary_size = 512

        with tf.variable_scope("fc1") as scope:
            W1 = tf.Variable(tf.random_normal([self.concat_size, intermediary_size], stddev=1e-3), name="w_fc")
            b1 = tf.Variable(tf.zeros([intermediary_size]), name="b_fc")

            z1 = tf.matmul(concat, W1) + b1

            if self.config.batch_norm:
                epsilon = 1e-3
                batch_mean1, batch_var1 = tf.nn.moments(z1, [0])
                scale1, beta1 = tf.Variable(tf.ones([intermediary_size])), tf.Variable(tf.zeros([intermediary_size]))
                z1 = tf.nn.batch_normalization(z1, batch_mean1, batch_var1, beta1, scale1, epsilon)

            fc1 = tf.nn.dropout(self.activation(z1), keep_prob=self.keep_prob)


        with tf.variable_scope("fc2") as scope:
            W2 = tf.Variable(tf.random_normal([intermediary_size, 1], stddev=1e-3), name="w_fc")
            b2 = tf.Variable(tf.zeros([1]), name="b_fc")

            z2 = tf.matmul(fc1, W2) + b2

            if self.config.batch_norm:
                epsilon = 1e-3
                batch_mean2, batch_var2 = tf.nn.moments(z2, [0])
                scale2, beta2 = tf.Variable(tf.ones([2])), tf.Variable(tf.zeros([2]))
                z2 = tf.nn.batch_normalization(z2, batch_mean2, batch_var2, beta2, scale2, epsilon)

            self.fc2 = tf.reshape(z2,[-1])


        ### Evaluation

        self.y_cos = self.activation(self.fc2)
        self.y_pre = tf.to_int32((tf.sign(self.y_cos*2-1)+1)/2)


    def build_loss_optimizer(self):
        ### Loss
        self.cross = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.fc2)
        # self.cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc2)
        self.losses = tf.reduce_mean(self.cross)

        ### Optimizer
        ### Optimizer
        if self.config.lr_decay == True:
            self.lr = tf.train.exponential_decay(learning_rate=self.config.learning_rate, global_step=self.global_step,
                                            decay_steps=1000, decay_rate=0.9, staircase=True)  # 每隔decay_steps步，lr=learning_rate*decay_rate, 比如global_step = n*decay_steps, lr = lr=learning_rate*(decay_rate)^n
        else:
            self.lr = tf.constant(self.config.learning_rate)
        with tf.variable_scope("train_step") as scope:
            if self.config.op_method == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.op_method == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.opt = optimizer.minimize(self.losses, global_step=self.global_step)


        # correct_prediction_inf = tf.equal(tf.argmax(self.fc2, 1), self.y)
        # self.accuracy_inf = tf.reduce_mean(tf.cast(correct_prediction_inf, tf.float32))
        #

    def train(self, batch_train_g, max_steps, save_path, save_every_n, log_every_n, val_g):

        with self.session as sess:
            # Train network
            # new_state = sess.run(self.initial_state)
            for q, q_len, r, r_len, y in batch_train_g:

                start = time.time()
                feed = {self.q1: q,
                        self.l1: q_len,
                        self.q2: r,
                        self.l2: r_len,
                        self.y: y,
                        self.keep_prob: self.config.train_keep_prob}
                batch_loss, _,fc2,y_cos,lr  = sess.run([self.losses, self.opt,self.fc2, self.y_cos,self.lr ], feed_dict=feed)
                end = time.time()

                # control the print lines
                if self.global_step.eval() % log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)),
                          'lr:{}'.format(lr))

                if (self.global_step.eval() % save_every_n == 0):
                    # self.saver.save(sess, os.path.join(save_path, 'model'), global_step=self.global_step)
                    y_pres = np.array([])
                    y_coss = np.array([])
                    y_s = np.array([])
                    for q, q_len, r, r_len, y in val_g:
                        feed = {self.q1: q,
                                self.l1: q_len,
                                self.q2: r,
                                self.l2: r_len,
                                self.y: y,
                                self.keep_prob: 1}
                        y_pre, y_cos = sess.run([self.y_pre, self.y_cos], feed_dict=feed)
                        y_pres = np.append(y_pres, y_pre)
                        y_coss = np.append(y_coss, y_cos)
                        y_s = np.append(y_s, y)
                    # 计算预测准确率
                    from sklearn.metrics import log_loss
                    y_coss[y_coss == 1] = 0.999999
                    logloss = log_loss(y_s, y_coss, eps=1e-15)
                    print('val lens:',len(y_s))
                    print('logloss:{:.4f}...'.format(logloss),
                          'best:{:.4f}'.format(self.global_loss.eval()),
                          "accuracy:{:.2f}%.".format((y_s == y_pres).mean() * 100))

                    if logloss < self.global_loss.eval():
                        print('save best model...')
                        update = tf.assign(self.global_loss, logloss)  # 更新最优值
                        sess.run(update)
                        self.saver.save(sess, os.path.join(save_path, 'model'), global_step=self.global_step)

                if self.global_step.eval() >= max_steps:
                    break
            # self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)


    def test(self, batch_generator, model_path ):
        with self.session as sess:
            q, q_len, r, r_len = batch_generator
            feed = {self.q1: q,
                    self.l1: q_len,
                    self.q2: r,
                    self.l2: r_len,
                    self.keep_prob: 1}
            y_pre, y_cos = sess.run([self.y_pre, self.y_cos,], feed_dict=feed)


            def make_submission(predict_prob):
                with open(model_path+'/submission.csv', 'a+') as file:
                    # file.write(str('y_pre') + '\n')
                    for line in predict_prob:
                        if line==1:
                            line = 0.99999
                        file.write(str(line) + '\n')
                file.close()
            make_submission(y_cos)
            print('...............................................................')

    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
