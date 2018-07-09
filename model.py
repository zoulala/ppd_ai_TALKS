from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os



class Model(object):
    '''seg_word ---> bilstm --> state.h-->  cos -->([0,1])
    restult: 0.530765
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

        # cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
        # cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)
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
        # self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
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
                we1 = tf.nn.dropout(we1, keep_prob=self.keep_prob)
                we2 = tf.nn.dropout(we2, keep_prob=self.keep_prob)

        ### ENCODER
        ### Shared layer
        with tf.variable_scope("bilstm") as scope:
            lstm1, state1 = self.bilstm(we1, self.l1)
            scope.reuse_variables()
            lstm2, state2 = self.bilstm(we2, self.l2)
            scope.reuse_variables()

        ### Features1
        state1_fw = state1[0]
        state1_bw = state1[1]
        state1_h_concat = tf.concat(values=[state1_fw.h, state1_bw.h], axis=1)

        state2_fw = state2[0]
        state2_bw = state2[1]
        state2_h_concat = tf.concat(values=[state2_fw.h, state2_bw.h], axis=1)

        flat1 = state1_h_concat
        flat2 = state2_h_concat

        # 转换矩阵
        with tf.variable_scope('bilinar_regression'):
             self.W = tf.get_variable("bilinear_W",shape=[self.config.hidden_size*2, self.config.hidden_size*2],
                                      initializer=tf.truncated_normal_initializer())

        # self.flat2 = tf.matmul(self.flat2, self.W)

        self.fc2 = self.get_cosine_similarity(flat1, flat2)

        ### Evaluation
        self.y_cos = self.fc2
        self.y_pre = tf.to_int32((tf.sign(self.y_cos * 2 - 1) + 1) / 2)

    @staticmethod
    def get_cosine_similarity(q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cosSim = tf.div(mul, tf.multiply(q1, a1))
        return cosSim

    def build_loss_optimizer(self):
        ### Loss
        self.cross = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.fc2)
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
                batch_loss, _,fc2,y_cos,lr= sess.run([self.losses, self.opt,self.fc2, self.y_cos,self.lr], feed_dict=feed)
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
            y_pre,y_cos = sess.run([self.y_pre, self.y_cos], feed_dict=feed)


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


from gensim import matutils
class WordVec():

    def __init__(self,embeddings):
        self.embeddings = embeddings

    def sens_to_embed(self,query_seqs):

        embed_query_seqs = tf.nn.embedding_lookup(self.embeddings, query_seqs)
        with tf.Session() as sess:
            embed_query_seqs = sess.run(embed_query_seqs)
            embed_query_seqs = embed_query_seqs.sum(axis=1)

            for i in range(embed_query_seqs.shape[0]):
                embed_query_seqs[i] = matutils.unitvec(embed_query_seqs[i])  # 单位圆化：模为1
        return embed_query_seqs





if __name__=="__main__":

    from sklearn.metrics import log_loss

    logloss = log_loss([0,1,1,1,1], [0, 1, 1, 1, 1], eps=1e-15)
    print('logloss:', logloss)

    from read_utils import TextConverter,val_samples_generator

    data_path,save_path = 'data','process_data'
    converter = TextConverter(data_path, save_path, 20)
    embeddings = converter.embeddings
    ww = WordVec(embeddings)



    val_samples = converter.load_obj(os.path.join(save_path, 'train_word.pkl'))
    val_g = val_samples_generator(val_samples[40000:80000])

    q_val,q_len,r_val,r_len,y = val_g

    embed_query_seqs = ww.sens_to_embed(q_val)
    embed_respones_seqs = ww.sens_to_embed(r_val)

    assert embed_query_seqs.shape[0]==embed_respones_seqs.shape[0],'not equal'

    n = embed_query_seqs.shape[0]

    print('start dot.')
    y_pre = []
    for i in range(n):
        nd = np.dot(embed_query_seqs[i], embed_respones_seqs[i])

        if nd > 0.9999999:
            nd = 1.0

        nd = (nd+1)/2

        if nd>0.65:
            y_pre.append(1)
        else:
            y_pre.append(0)
        # y_pre.append(nd)



    if len(y) == len(y_pre):

        # 计算预测准确率（百分比）
        print( "Predictions have an accuracy of {:.2f}%.".format((y == np.array(y_pre)).mean() * 100))

    else:
        print( "Number of predictions does not match number of outcomes!")


    logloss = log_loss(y, y_pre, eps=1e-15)
    print('logloss:', logloss)




