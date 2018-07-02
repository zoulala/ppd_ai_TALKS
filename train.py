import tensorflow as tf
from read_utils import TextConverter, batch_generator,samples_clearn,val_samples_generator
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
    test_args.add_argument('--train_keep_prob', type=float, default=0.7,help='dropout rate during training')
    test_args.add_argument('--max_steps', type=int, default=100000,help='max steps to train')
    test_args.add_argument('--save_every_n', type=int, default=1000,help='save the model every n steps')
    test_args.add_argument('--log_every_n', type=int, default=100,help='log to the screen every n steps')
    test_args.add_argument('--fc_activation', type=str, default='sigmoid', help='funciton of activated')
    test_args.add_argument('--feats', type=str, default='all', help='features of query')
    test_args.add_argument('--batch_norm', type=bool, default=False, help='standardization')
    test_args.add_argument('--op_method', type=str, default='adam', help='method of optimizer')
    test_args.add_argument('--checkpoint_path', type=str, default='models/thoth3/', help='checkpoint path')
    test_args.add_argument('--lr_decay', type=bool, default=False, help='standardization')




    return parser.parse_args(args)


## thoth 问答
args_in = '--file_name hd300all_thoth3 ' \
          '--num_steps 20 ' \
          '--batch_size 64 ' \
          '--learning_rate 0.001 ' \
          '--use_embedding Ture ' \
          '--hidden_size 300 ' \
          '--fc_activation sigmoid ' \
          '--op_method adam ' \
          '--max_steps 200000'.split()

FLAGS = parseArgs(args_in)



def main(_):
    word_char = 'word'  # 'word' or 'char'
    print('use word or char:',word_char)

    FLAGS.file_name = word_char+'_'+FLAGS.file_name
    print('model_path:',FLAGS.file_name)

    model_path = os.path.join('models', FLAGS.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    if FLAGS.file_name[-1] == '2':
        from model2 import Model
    elif FLAGS.file_name[-1] == '3':
        from model3 import Model
    elif FLAGS.file_name[-1] == '4':
        from model4 import Model
    elif FLAGS.file_name[-1] == '5':
        from model5 import Model
    else:
        from model1 import Model

    data_path,save_path = 'data','process_data'

    converter = TextConverter(word_char, data_path, save_path,  FLAGS.num_steps)
    embeddings = converter.embeddings

    if word_char == 'word':
        train_pkl = 'train_word.pkl'
        val_pkl = 'val_word.pkl'
    if word_char == 'char':
        train_pkl = 'train_char.pkl'
        val_pkl = 'val_char.pkl'

    train_samples = converter.load_obj(os.path.join(save_path, train_pkl))
    train_g = batch_generator(train_samples, FLAGS.batch_size)

    val_samples = converter.load_obj(os.path.join(save_path, val_pkl))
    val_g = val_samples_generator(val_samples[:5000])


    print('use embeding:',FLAGS.use_embedding)
    print('vocab size:',converter.vocab_size)


    model = Model(converter.vocab_size,FLAGS,test=False, embeddings=embeddings)

    # 继续上一次模型训练
    FLAGS.checkpoint_path = tf.train.latest_checkpoint(model_path)
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