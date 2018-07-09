
from tqdm import tqdm
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import pickle
import random

def samples_clearn(samples):
    cl_samples = [sample for sample in samples if sample[4]==1]
    return cl_samples

def val_samples_generator(samples):
    val_g = []
    n = len(samples)
    batchsize = 4000
    for i in range(0,n,batchsize):
        batch_samples = samples[i:i + batchsize]
        batch_q = []
        batch_q_len = []
        batch_r = []
        batch_r_len = []
        batch_y = []
        for sample in batch_samples:
            batch_q.append(sample[0])
            batch_q_len.append(sample[1])
            batch_r.append(sample[2])
            batch_r_len.append(sample[3])
            batch_y.append(sample[4])

        batch_q = np.array(batch_q)
        batch_q_len = np.array(batch_q_len)
        batch_r = np.array(batch_r)
        batch_r_len = np.array(batch_r_len)
        batch_y = np.array(batch_y)
        val_g.append( (batch_q,batch_q_len,batch_r,batch_r_len,batch_y))
    return val_g


def test_samples_generator(samples):
    batch_q = []
    batch_q_len = []
    batch_r = []
    batch_r_len = []
    for sample in samples:
        batch_q.append(sample[0])
        batch_q_len.append(sample[1])
        batch_r.append(sample[2])
        batch_r_len.append(sample[3])

    batch_q = np.array(batch_q)
    batch_q_len = np.array(batch_q_len)
    batch_r = np.array(batch_r)
    batch_r_len = np.array(batch_r_len)

    return batch_q,batch_q_len,batch_r,batch_r_len

def batch_generator(samples, batchsize):
    '''产生训练batch样本'''
    n_samples = len(samples)
    n_batches = int(n_samples/batchsize)
    n = n_batches * batchsize
    while True:
        random.shuffle(samples)  # 打乱顺序
        for i in range(0, n, batchsize):
            batch_samples = samples[i:i+batchsize]
            batch_q = []
            batch_q_len = []
            batch_r = []
            batch_r_len = []
            batch_y = []
            for sample in batch_samples:
                batch_q.append(sample[0])
                batch_q_len.append(sample[1])
                batch_r.append(sample[2])
                batch_r_len.append(sample[3])
                batch_y.append(sample[4])

            batch_q = np.array(batch_q)
            batch_q_len = np.array(batch_q_len)
            batch_r = np.array(batch_r)
            batch_r_len = np.array(batch_r_len)
            batch_y = np.array(batch_y)
            yield batch_q,batch_q_len,batch_r,batch_r_len,batch_y

class TextConverter():
    def __init__(self,word_char, data_path, save_path,  max_steps=20):
        self.max_steps = max_steps
        self.word_char = word_char

        if self.word_char == 'word':
            if os.path.exists(os.path.join(save_path, 'train_word.pkl')) is False:
                self.initialize(data_path, save_path)
            self.embeddings = self.load_embedding(os.path.join(save_path, 'word_embed.npy'))
        elif self.word_char == 'char':
            if os.path.exists(os.path.join(save_path, 'train_char.pkl')) is False:
                self.initialize(data_path, save_path)
            self.embeddings = self.load_embedding(os.path.join(save_path, 'char_embed.npy'))
        else:
            raise Exception("Invalid value!", self.word_char)

        self.vocab_size = self.embeddings.shape[0]

    def initialize(self,data_path, save_path):
        ori_train_csv = pd.read_csv(os.path.join(data_path, 'train.csv'))
        thres = int(len(ori_train_csv) * 0.9)
        self.train_csv = ori_train_csv[:thres]
        self.val_csv = ori_train_csv[thres:]
        self.test_csv = pd.read_csv(os.path.join(data_path, 'test.csv'))

        self.question_to_cleaned(os.path.join(data_path, 'question.csv'),os.path.join(data_path, 'question_cleaned.csv'))
        self.q2w, self.q2c = self.fn(os.path.join(data_path, 'question_cleaned.csv'))
        assert len(self.q2w) == len(self.q2c), "len(q2w): %d, len(q2c): %d" % (len(self.q2w), len(self.q2c))

        self.save_obj(self.q2w, os.path.join(save_path, 'q2w.pkl'))
        self.save_obj(self.q2c, os.path.join(save_path, 'q2c.pkl'))

        if self.word_char == 'word':
            self.embed_to_val(os.path.join(data_path, 'word_embed.txt'),os.path.join(save_path, 'word_embed.npy'))
            self.train_fn(self.train_csv, os.path.join(save_path, 'train_word.pkl'))
            self.test_fn(self.test_csv, os.path.join(save_path, 'test_word.pkl'))
            self.train_fn(self.val_csv, os.path.join(save_path, 'val_word.pkl'))

        if self.word_char == 'char':
            self.embed_to_val(os.path.join(data_path, 'char_embed.txt'),os.path.join(save_path, 'char_embed.npy'))
            self.train_fn(self.train_csv, os.path.join(save_path, 'train_char.pkl'))
            self.test_fn(self.test_csv, os.path.join(save_path, 'test_char.pkl'))
            self.train_fn(self.val_csv, os.path.join(save_path, 'val_char.pkl'))


    def glance(self,d, n=1):
        return dict(itertools.islice(d.items(), 1))

    def fn(self, path):
        _q2w, _q2c = {}, {}

        with open(path) as f:
            next(f)
            for line in f:
                l_split = line.split(',')
                qid, words, chars = l_split

                words_sp = words.split()
                chars_sp = chars.split()

                _q2w[qid] = words_sp
                _q2c[qid] = chars_sp

        return _q2w, _q2c


    def question_to_cleaned(self, open_path,save_path):
        with open(open_path) as f:
            st = f.read()
            st = st.replace('W0', '')
            st = st.replace('L0', '')
            st = st.replace('W', '')
            st = st.replace('L', '')
        with open(save_path, 'w') as f:
            f.write(st)


    def save_obj(self,obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self,path):
        with open(path, 'rb') as f:
            return pickle.load(f)


    def embed_to_val(self,open_path,save_path):
        embed_vals = []
        with open(open_path) as f:
            for line in f:
                line_sp = line.split()
                embed_vals.append([float(num) for num in line_sp[1:]])
        embed_vals = np.asarray(embed_vals, dtype=np.float32)
        self.PAD_INT = embed_vals.shape[0]
        zeros = np.zeros((1,300), dtype=np.float32)
        embed_vals = np.concatenate([embed_vals, zeros])
        np.save(save_path, embed_vals)


    def load_embedding(self,path):
        embedding = np.load(path)
        return embedding


    def fn1(self,str_li, int_li):
        str_len = len(str_li)
        if str_len > self.max_steps:
            str_len = self.max_steps
        for i, s in enumerate(str_li[:self.max_steps]):
            int_li[i] = int(str_li[i])

        return str_len



    def train_fn(self, csv, path):
        samples = []
        for arr_line in tqdm(csv.values, total=len(csv), ncols=70):
            q1_id_int, q2_id_int = [self.PAD_INT] * self.max_steps, [self.PAD_INT] * self.max_steps

            label, q1_id, q2_id = arr_line
            if self.word_char == 'word':
                q1_len = self.fn1(self.q2w[q1_id], q1_id_int)
                q2_len = self.fn1(self.q2w[q2_id], q2_id_int)
            if self.word_char == 'char':
                q1_len = self.fn1(self.q2c[q1_id], q1_id_int)
                q2_len = self.fn1(self.q2c[q2_id], q2_id_int)

            samples.append((q1_id_int,q1_len,q2_id_int,q2_len,label))
        self.save_obj(samples, path)

    def test_fn(self, csv, path):
        samples = []
        for arr_line in tqdm(csv.values, total=len(csv), ncols=70):
            q1_id_int, q2_id_int = [self.PAD_INT] * self.max_steps, [self.PAD_INT] * self.max_steps

            q1_id, q2_id = arr_line
            if self.word_char == 'word':
                q1_len = self.fn1(self.q2w[q1_id], q1_id_int)
                q2_len = self.fn1(self.q2w[q2_id], q2_id_int)
            if self.word_char == 'char':
                q1_len = self.fn1(self.q2c[q1_id], q1_id_int)
                q2_len = self.fn1(self.q2c[q2_id], q2_id_int)

            samples.append((q1_id_int, q1_len, q2_id_int, q2_len))
        self.save_obj(samples, path)



if __name__=="__main__":
    pro_data = TextConverter('word','data','process_data1',26)

    print(pro_data.PAD_INT)

