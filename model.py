from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os


class DualBiLSTM():
    '''
    分别对问句和答案进行encode
    '''
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, test=False, train_keep_prob=0.5, use_embedding=False, embedding_size=300):
        # if test is True:
        #     batch_size = 1

        self.num_classes = num_classes  # 网络输出分类数量，（文字字典大小）
        self.batch_size = batch_size
        self.num_steps = num_steps  # 序列时间维度
        self.lstm_size = lstm_size  # 隐层神经元个数
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip  # 梯度下降参数最大限度
        self.train_keep_prob = train_keep_prob  # 训练以一定概率进行
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size  # 嵌入词向量大小，即输入节点个数


        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        if test is False:
            self.build_loss()
            self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):

            self.query_seqs = tf.placeholder(tf.int32, [None, self.num_steps], name='query')
            self.query_length = tf.placeholder(tf.int32, [None], name='query_length')

            self.response_seqs = tf.placeholder(tf.int32, [None, self.num_steps], name='response')
            self.response_length = tf.placeholder(tf.int32, [None], name='response_length')

            # self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps),name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=[self.batch_size], name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 词嵌入层
            if self.use_embedding is False:
                self.lstm_query_seqs = tf.one_hot(self.query_seqs, depth=self.num_classes)   # 独热编码[1,2,3] depth=5 --> [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]，此时的输入节点个数为num_classes
                self.lstm_response_seqs = tf.one_hot(self.response_seqs, depth=self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_query_seqs = tf.nn.embedding_lookup(embedding, self.query_seqs)  # 词嵌入[1,2,3] --> [[3,...,4],[0.7,...,-3],[6,...,9]],embeding[depth*embedding_size]=[[0.2,...,6],[3,...,4],[0.7,...,-3],[6,...,9],[8,...,-0.7]]，此时的输入节点个数为embedding_size
                    self.lstm_response_seqs = tf.nn.embedding_lookup(embedding, self.response_seqs)

    def build_lstm(self):

        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0)
        if self.train_keep_prob is not None:
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.train_keep_prob)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.train_keep_prob)


        # self.initial_state1 = cell.zero_state(self.batch_size, tf.float32)
        # self.initial_state2 = cell.zero_state(self.batch_size, tf.float32)
        # self.initial_state1 = cell.zero_state(1, tf.float32)
        # self.initial_state2 = cell.zero_state(9212, tf.float32)

        # 通过dynamic_rnn对cell展开时间维度
        query_output, self.query_state= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,
                                                        inputs=self.lstm_query_seqs,
                                                        sequence_length=self.query_length,
                                                        # initial_state=self.initial_state1,  # 可有可无，自动为0状态
                                                        time_major=False,
                                                        dtype=tf.float32)
        response_output, self.response_state= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,
                                                        inputs=self.lstm_response_seqs,
                                                        sequence_length=self.response_length,
                                                        # initial_state=self.initial_state1,  # 可有可无，自动为0状态
                                                        time_major=False,
                                                        dtype=tf.float32)
        query_c_fw, query_h_fw = self.query_state[0]
        query_c_bw, query_h_bw = self.query_state[1]
        response_c_fw, response_h_fw = self.response_state[0]
        response_c_bw, response_h_bw = self.response_state[1]

        self.query_h_state = tf.concat([query_h_fw, query_h_bw],axis=1)

        self.response_h_state = tf.concat([response_h_fw, response_h_bw],axis=1)

        # 转换矩阵
        with tf.variable_scope('bilinar_regression'):
             self.W = tf.get_variable("bilinear_W",shape=[self.lstm_size*2, self.lstm_size*2],
                                      initializer=tf.truncated_normal_initializer())

        # 训练阶段, 使用batch内其他样本的response作为negative response
        self.response_matul_state = tf.matmul(self.response_h_state, self.W)
        self.logits = tf.matmul(a=self.query_h_state, b=self.response_matul_state, transpose_b=True)
        # self.diag_logits = tf.diag_part(self.logits)  # 获取对角线元素

    def build_loss(self):
        with tf.name_scope('loss'):
            self.diag_targets = tf.matrix_diag(self.targets)  # 生成对角矩阵
            # self.losses = tf.losses.softmax_cross_entropy(onehot_labels=self.diag_targets, logits=self.logits)
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.diag_targets)
            self.mean_loss = tf.reduce_mean(self.losses, name="mean_loss")  # batch样本的平均损失


    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            # new_state = sess.run(self.initial_state)
            for q, q_len, r, r_len, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.query_seqs: q,
                        self.query_length: q_len,
                        self.response_seqs: r,
                        self.response_length: r_len,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob}
                batch_loss,losses, _ = sess.run([self.mean_loss,self.losses, self.optimizer], feed_dict=feed)
                end = time.time()

                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                    # print(losses)
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            # self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)


    def test(self, batch_generator ):
        sess = self.session
        for q, q_len, r, r_len, y in batch_generator:
            feed = {self.query_seqs: q[0].reshape(-1,26),
                    self.query_length: q_len[0].reshape(1),
                    self.response_seqs: r,
                    self.response_length: r_len,
                    self.keep_prob: 1.}
            logits = sess.run(tf.nn.softmax(self.logits), feed_dict=feed)
            print('概率：', logits)
