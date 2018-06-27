import tensorflow as tf
from read_utils import TextConverter, batch_generator,samples_clearn,val_samples_generator
from model import DualBiLSTM
from model2 import Model
import os
import argparse  # 用于分析输入的超参数


def parseArgs(args):
    """
    Parse 超参数
    Args:
        args (list<stir>): List of arguments.
    """

    parser = argparse.ArgumentParser()
    test_args = parser.add_argument_group('test超参数')
    test_args.add_argument('--file_name', type=str, default='default',help='name of the model')
    test_args.add_argument('--batch_size', type=int, default=100,help='number of seqs in one batch')
    test_args.add_argument('--num_steps', type=int, default=100,help='length of one seq')
    test_args.add_argument('--hidden_size', type=int, default=128,help='size of hidden state of lstm')
    test_args.add_argument('--num_layers', type=int, default=2,help='number of lstm layers')
    test_args.add_argument('--use_embedding', type=bool, default=False,help='whether to use embedding')
    test_args.add_argument('--embedding_size', type=int, default=128,help='size of embedding')
    test_args.add_argument('--learning_rate', type=float, default=0.001,help='learning_rate')
    test_args.add_argument('--train_keep_prob', type=float, default=1,help='dropout rate during training')
    test_args.add_argument('--input_file', type=str, default='',help='utf8 encoded text file')
    test_args.add_argument('--max_steps', type=int, default=100000,help='max steps to train')
    test_args.add_argument('--save_every_n', type=int, default=1000,help='save the model every n steps')
    test_args.add_argument('--log_every_n', type=int, default=10,help='log to the screen every n steps')
    test_args.add_argument('--max_vocab', type=int, default=8000,help='max char number')
    test_args.add_argument('--sheetname', type=str, default='default',help='name of the model')
    test_args.add_argument('--fc_activation', type=str, default='sigmoid', help='funciton of activated')
    test_args.add_argument('--feats', type=str, default='raw', help='features of query')
    test_args.add_argument('--batch_norm', type=bool, default=False, help='standardization')
    test_args.add_argument('--op_method', type=str, default='adam', help='method of optimizer')
    test_args.add_argument('--checkpoint_path', type=str, default='models/thoth/', help='checkpoint path')



    return parser.parse_args(args)


## thoth 问答
args_in = '--file_name thoth ' \
          '--num_steps 20 ' \
          '--batch_size 32 ' \
          '--learning_rate 0.001 ' \
          '--use_embedding Ture ' \
          '--hidden_size 128 ' \
          '--max_steps 200000'.split()

FLAGS = parseArgs(args_in)



def main(_):
    model_path = os.path.join('models', FLAGS.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)


    data_path,save_path = 'data','process_data'
    converter = TextConverter(data_path, save_path, FLAGS.num_steps)
    embeddings = converter.embeddings
    train_samples = converter.load_obj(os.path.join(save_path, 'train_word.pkl'))
    # samples = samples_clearn(samples)
    train_g = batch_generator(train_samples, FLAGS.batch_size)

    val_samples = converter.load_obj(os.path.join(save_path, 'val_word.pkl'))
    val_g = val_samples_generator(val_samples[:5000])

    print(FLAGS.use_embedding)
    print(converter.vocab_size)
    # model = DualBiLSTM(converter.vocab_size,
    #                  batch_size=FLAGS.batch_size,
    #                  num_steps=FLAGS.num_steps,
    #                  lstm_size=FLAGS.hidden_size,
    #                  num_layers=FLAGS.num_layers,
    #                  learning_rate=FLAGS.learning_rate,
    #                  train_keep_prob=FLAGS.train_keep_prob,
    #                  use_embedding=FLAGS.use_embedding,
    #                  embedding_size=FLAGS.embedding_size,
    #                  embeddings=embeddings
    #                  )

    model = Model(converter.vocab_size,FLAGS,test=False, embeddings=embeddings)

    # 继续训练
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    if FLAGS.checkpoint_path:
        model.load(FLAGS.checkpoint_path)

    model.train(train_g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                val_g
                )


if __name__ == '__main__':
    tf.app.run()