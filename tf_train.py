import tensorflow as tf
import numpy as np
from params import *
from tf_net import TasNet
from data_generator import DataGenerator
import os
from datetime import datetime
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
scratch_or_resume = False
seq_duration = seq_duration if scratch_or_resume else 2.0
if scratch_or_resume:
    seq_duration = seq_duration
    val_list = '/home/grz/data/SSSR/wsj0_tasnet/cv/raw_100-52825.pkl'
    trn_list = '/home/grz/data/SSSR/wsj0_tasnet/tr/raw_100-208965.pkl'
else:
    if seq_duration == 2.0:
        val_list = '/home/grz/data/SSSR/wsj0_tasnet/cv/raw_400-11338.pkl'
        trn_list = '/home/grz/data/SSSR/wsj0_tasnet/tr/raw_400-44956.pkl'
        batch_size = 30
    else:
        val_list = '/home/grz/data/SSSR/wsj0_tasnet/cv/raw_800-5486.pkl'
        trn_list = '/home/grz/data/SSSR/wsj0_tasnet/tr/raw_800-21789.pkl'
        batch_size = 16
    model_subpath = '100-e19.ckpt-32641'
    model_path = os.getcwd() + '/' + sum_dir + '/model/' + model_subpath

seq_len = int(seq_duration / sample_duration)


def train(max_epoch):

    # =============== PLACEHOLDER & MODEL DEFINITION ========================
    with tf.Graph().as_default():
        mixture = tf.placeholder(shape=[None, None, L], dtype=tf.float32, name='mixture')
        source = tf.placeholder(shape=[None, nspk, None, L], dtype=tf.float32, name='source')
        train_lr = tf.placeholder(shape=None, dtype=tf.float32, name='lr')

        print('>> Initializing model...')
        model = TasNet(batch_size=batch_size, seq_len=seq_len)
        est_source = model.build_network(mixture)
        # loss, min_mse, est_source, reorder_recon = model.MSE_objective(source=source,
        # est_source=est_source)
        loss, max_snr, est_source, _ = model.objective(est_source=est_source, source=source)
        train_op = model.train(loss, lr=train_lr)

        saver = tf.train.Saver(tf.global_variables())
        # summary_op = tf.summary.merge_all()
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        if scratch_or_resume:
            val_loss = []
            val_loss_min = np.inf
            step = 0
            last_epoch = 0
        else:
            saver.restore(sess, model_path)
            step = int(model_subpath.split(sep='-')[-1])
            last_epoch = int(model_subpath.split(sep='.')[0].split(sep='-')[-1][1:])
            print('load model from: ', model_path)

        # =============== SUMMARY & DATA ========================
        summary_dir = sum_dir + '/' + 'duration' + str(seq_duration)
        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)
            os.mkdir(summary_dir + '/train')
            os.mkdir(summary_dir + '/val')
        summary_writer_train = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
        summary_writer_val = tf.summary.FileWriter(summary_dir + '/val')

        # ----------------------------------- DATA ------------------------------------------
        # 1> generator for training set and validation set
        print('>> Loading data...')
        val_generator = DataGenerator(batch_size=batch_size, max_k=seq_len,
                                      name='val-generator')
        data_generator = DataGenerator(batch_size=batch_size, max_k=seq_len,
                                       name='train-generator')
        val_generator.load_data(val_list)
        data_generator.load_data(trn_list)

        train_lr_value = lr
        data_generator.epoch = last_epoch
        train_loss_ = []
        val_no_best = 0
        print('>> Training...  %s start' % datetime.now())
        while last_epoch <= max_epoch:
            step += 1
            start_time = time.time()

            data_batch = data_generator.gen_batch()
            loss_value, _, \
            sum1, sum2, sum3, \
            sum4, sum5, \
            sum6, sum7 = sess.run([loss, train_op,
                                    model.loss_summary, model.snr_summary, model.summary_conv,
                                    model.summary_gate, model.summary_lstm_out,
                                    model.summary_layer_norm_mix, model.summary_B],
                                   feed_dict={mixture: np.array(data_batch['mix']).astype(np.float32),
                                              source: np.array(data_batch['s']).astype(np.float32),
                                              train_lr: train_lr_value})
            for sum_idx in range(1, 8):
                eval('summary_writer_train.add_summary(sum' + str(sum_idx) + ', step)')
            train_loss_.append(loss_value)
            duration = time.time() - start_time
            if np.isnan(loss_value):
                print('NAN loss: epoch %d step %d' % (last_epoch, step))

            if step % display_freq == 0:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = '%s: step %d, loss = %.5f, (%.1f sp/s; %.3f s/batch, epoch %d)'
                print(format_str % (get_time_str(), step, sum(train_loss_) / len(train_loss_),
                                    examples_per_sec, sec_per_batch,
                                    data_generator.epoch))
                train_loss_ = []

            # ----------------------------------- VALIDATION ------------------------------------------
            if last_epoch != data_generator.epoch:
                # doing validation every training epoch
                print('>> Current epoch: ', last_epoch, ', doing validation')
                val_epoch = val_generator.epoch
                count, loss_sum, sum1, sum2 = 0, 0, '', ''
                # average the validation loss
                while val_epoch == val_generator.epoch:
                    count += 1
                    data_batch = val_generator.gen_batch()
                    loss_value, sum1, sum2 = sess.run([loss, model.loss_summary, model.snr_summary],
                                                feed_dict={mixture: np.array(data_batch['mix']).astype(np.float32),
                                                           source: np.array(data_batch['s']).astype(np.float32)})
                    loss_sum += loss_value
                summary_writer_val.add_summary(sum1, step)
                summary_writer_val.add_summary(sum2, step)
                val_loss_sum = (loss_sum / count)
                val_loss.append(val_loss_sum)
                format_str = 'validation: loss = %.5f'
                print(format_str % (val_loss_sum))
                np.array(val_loss).tofile(sum_dir + '/loss/val_' + str(seq_len))
                if val_loss_sum < val_loss_min:
                    print('# train_net: saving model at step %d because of minimum validation loss' % step)
                    save_model(sess, saver, last_epoch, step)
                    val_loss_min = val_loss_sum
                    val_no_best = 0
                else:
                    val_no_best += 1
                    if val_no_best == 3:
                        train_lr_value = train_lr_value / 2
                        print('# no improvement in 3 epochs, reduce learning rate to:',
                              train_lr_value)
                        val_no_best = 0
                    if val_no_best >= 10:
                        print('# Early stop! ')
                        break
            last_epoch = data_generator.epoch
            if last_epoch == max_epoch:
                save_model(sess, saver, last_epoch, step)
                print('reach max training epoch.')
                return

        sess.close()


def get_time_str():
    time = datetime.now()
    timestr = '%02d%02d%02d-%02d%02d%02d' % (time.year, time.month, time.day,
                                             time.hour, time.minute, time.second)
    return timestr


def save_model(sess, saver, last_epoch, step):
    checkpoint_path = os.path.join(sum_dir + '/model/',
                                   str(seq_len) + '-e' + str(last_epoch) + '.ckpt')
    saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train(100)
