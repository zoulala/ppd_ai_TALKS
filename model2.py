from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os



class Model(object):
    '''seg_word ---> bilstm --> output--> maxpooling -->  h*w+b --> h*w+b -->([0,1])
    result:0.635849
    '''
    def __init__(self, num_classes, config, test=False, embeddings=None):
        self.num_classes = num_classes
        self.config = config
        self.embeddings = embeddings

        tf.reset_default_graph()
        self.build_nn()
        if test is False:
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

        self.y = tf.placeholder( tf.int64, shape=[None], name="is_duplicate")

        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(1, dtype=tf.float32,trainable=False, name="global_loss")

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

        ### ENCODER
        ### Shared layer
        with tf.variable_scope("bilstm") as scope:
            lstm1, state1 = self.bilstm(we1, self.l1)
            scope.reuse_variables()
            lstm2, state2 = self.bilstm(we2, self.l2)
            scope.reuse_variables()


        ### INFERENCE
        ### Max pooling
        max_pool = tf.contrib.keras.layers.GlobalMaxPool1D()
        lstm1_pool = max_pool(lstm1)  # [batch, n_steps, embed_size]-->[batch, embed_size],选取n_steps中最大的保留。如[[[1,2,3,4],[2,-1,0,6],[7,0,-2,1]],...,[[1,2,3,4],[2,-1,0,9],[0,1,-1,4]]]-->[7,2,3,6],...,[2,2,3,9]
        lstm2_pool = max_pool(lstm2)

        ### Features
        flat1 = tf.contrib.layers.flatten(lstm1_pool)  #作用reshape：flattened = tf.reshape(x, [tf.shape(x)[0], -1])，最终维度是2:[batch,n]
        flat2 = tf.contrib.layers.flatten(lstm2_pool)
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
            W2 = tf.Variable(tf.random_normal([intermediary_size, 2], stddev=1e-3), name="w_fc")
            b2 = tf.Variable(tf.zeros([2]), name="b_fc")

            z2 = tf.matmul(fc1, W2) + b2

            if self.config.batch_norm:
                epsilon = 1e-3
                batch_mean2, batch_var2 = tf.nn.moments(z2, [0])
                scale2, beta2 = tf.Variable(tf.ones([2])), tf.Variable(tf.zeros([2]))
                z2 = tf.nn.batch_normalization(z2, batch_mean2, batch_var2, beta2, scale2, epsilon)

            self.fc2 = z2




    def build_loss_optimizer(self):
        ### Loss
        self.cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc2)
        self.losses = tf.reduce_mean(self.cross)

        ### Optimizer
        with tf.variable_scope("train_step") as scope:
            if self.config.op_method == "adam":
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            elif self.config.op_method == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
            self.opt = optimizer.minimize(self.losses, global_step=self.global_step)

        ### Evaluation
        self.y_pre = tf.argmax(self.fc2, 1)
        self.y_cos = tf.nn.softmax(logits=self.fc2)[:,-1]
        # correct_prediction_inf = tf.equal(tf.argmax(self.fc2, 1), self.y)
        # self.accuracy_inf = tf.reduce_mean(tf.cast(correct_prediction_inf, tf.float32))
        #

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n, val_g):

        with self.session as sess:
            # Train network
            # new_state = sess.run(self.initial_state)
            for q, q_len, r, r_len, y in batch_generator:

                start = time.time()
                feed = {self.q1: q,
                        self.l1: q_len,
                        self.q2: r,
                        self.l2: r_len,
                        self.y: y,
                        self.keep_prob: self.config.train_keep_prob}
                batch_loss, _,y_pre, y_cos = sess.run([self.losses, self.opt,self.y_pre, self.y_cos], feed_dict=feed)
                end = time.time()

                # control the print lines
                if self.global_step.eval() % log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (self.global_step.eval() % save_every_n == 0):
                    # self.saver.save(sess, os.path.join(save_path, 'model'), global_step=self.global_step)

                    q, q_len, r, r_len, y = val_g
                    feed = {self.q1: q,
                            self.l1: q_len,
                            self.q2: r,
                            self.l2: r_len,
                            self.y: y,
                            self.keep_prob: 1}
                    batch_loss,  y_pre, y_cos = sess.run([self.losses,  self.y_pre, self.y_cos], feed_dict=feed)
                    print(y[:30])
                    print(y_pre[:30])
                    print(y_cos[:10])
                    if len(y) == len(y_pre):
                        # 计算预测准确率（百分比）
                        print("Predictions have an accuracy of {:.2f}%.".format((y == np.array(y_pre)).mean() * 100))
                    else:
                        print("Number of predictions does not match number of outcomes!")

                    from sklearn.metrics import log_loss
                    y_cos[y_cos == 1] = 0.999999
                    logloss = log_loss(y, y_cos, eps=1e-15)
                    print('logloss:',logloss)

                    if logloss < self.global_loss.eval():
                        update = tf.assign(self.global_loss, logloss)
                        sess.run(update)
                        self.saver.save(sess, os.path.join(save_path, 'model'), global_step=self.global_step)



                if self.global_step.eval() >= max_steps:
                    break
            # self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)



    def test(self, batch_generator,model_path ):
        with self.session as sess:
            q, q_len, r, r_len = batch_generator
            feed = {self.q1: q,
                    self.l1: q_len,
                    self.q2: r,
                    self.l2: r_len,
                    self.keep_prob: 1}
            y_pre,y_cos = sess.run([self.y_pre, self.y_cos], feed_dict=feed)

            print(y_pre[:30])
            print(y_cos[:30])

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
