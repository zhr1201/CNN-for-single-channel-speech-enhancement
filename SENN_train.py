'''
Main script for training the net
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import ipdb
import numpy as np
import tensorflow as tf
import SENN
import audio_reader

LR = 0.00001

FLAGS = tf.app.flags.FLAGS

# store the check points
tf.app.flags.DEFINE_string(
    'train_dir',
    '/home/nca/Downloads/speech_enhencement_large/speech_enhencement2/SENN2',
    """Directory where to write event logs """)
# write summary about the loss and etc.
tf.app.flags.DEFINE_string(
    'sum_dir',
    '/home/nca/Downloads/speech_enhencement_large/speech_enhencement2/SENN2/sum2',
    """Directory where to write summary """)
# noise directory
tf.app.flags.DEFINE_string(
    'noise_dir',
    '/media/nca/data/raw_data/Nonspeech_train/',
    # '/home/nca/Downloads/raw_data/Nonspeech_train/',
    """Directory where to load noise """)
# speech directory
tf.app.flags.DEFINE_string(
    'speech_dir',
    '/media/nca/data/raw_data/speech_train/',
    # '/home/nca/Downloads/raw_data/speech_train/',
    """Directory where to load speech """)
# validation noise directory
tf.app.flags.DEFINE_string(
    'val_noise_dir',
    '/media/nca/data/raw_data/Nonspeech_test/',
    # '/home/nca/Downloads/raw_data/Nonspeech_test/',
    """Directory where to load noise """)
# validation speech directory
tf.app.flags.DEFINE_string(
    'val_speech_dir',
    '/media/nca/data/raw_data/speech_test/',
    # '/home/nca/Downloads/raw_data/speech_test/',
    """Directory where to load noise """)
tf.app.flags.DEFINE_integer('max_steps', 2000000000,
                            """Number of batches to run.""")

NFFT = 256  # number of fft points
NEFF = 129  # number of effective fft points
frame_move = 64  # hop size
batch_size = 128
N_IN = 8  # number of frames presented to the net
N_OUT = 1  # output frame number
validation_samples = 848824  # total numbers of the validation set
batch_of_val = np.floor(validation_samples / batch_size)
# after all the batches, dequeue the left to make sure
# all the samples in the validation set are the same
val_left_to_dequeue = validation_samples - batch_of_val * batch_size
val_loss = np.zeros([1000000])


def train():
    coord = tf.train.Coordinator()

    # speech reader
    audio_rd = audio_reader.Audio_reader(
        FLAGS.speech_dir, FLAGS.noise_dir, coord, N_IN, NFFT,
        frame_move, is_val=False)

    # noise reader
    val_audio_rd = audio_reader.Audio_reader(
        FLAGS.val_speech_dir, FLAGS.val_noise_dir, coord, N_IN, NFFT,
        frame_move, is_val=False)

    # flag for validation or training
    is_val = tf.placeholder(dtype=tf.bool, shape=())

    # speech enhancement net
    SE_Net = SENN.SE_NET(
        batch_size, NEFF, N_IN, N_OUT)

    # raw data frames
    train_data_frames = audio_rd.dequeue(batch_size)

    val_data_frames = val_audio_rd.dequeue(batch_size)

    # select which to use in validation or training
    data_frames = tf.cond(
        is_val, lambda: val_data_frames, lambda: train_data_frames)

    # transform raw data into inputs for the nets
    # it is not done in preprocessing because it runs really fast
    # and we don't need to store all the mixed samples
    images, targets = SE_Net.inputs(data_frames)

    # infer the clean speech
    inf_targets = SE_Net.inference(images, is_train=True)

    loss = SE_Net.loss(inf_targets, targets)  # compute loss

    train_op = SE_Net.train(loss, LR)  # optimizer

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()

    sess = tf.Session()

    sess.run(init)

    audio_rd.start_threads(sess)  # start audio reading threads
    val_audio_rd.start_threads(sess)

    # tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(
        FLAGS.sum_dir,
        sess.graph)

    # to track the times of validation
    val_loss_id = 0
    for step in xrange(FLAGS.max_steps):

        start_time = time.time()
        _, loss_value = sess.run(
            [train_op, loss], feed_dict={is_val: False})
        # images_batch, targets_batch, inf_batch, _, loss_value = sess.run(
        #     [images, targets, inf_targets, train_op, loss], feed_dict={is_val: False})
        # ipdb.set_trace()
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        # display training loss every 100 steps
        if step % 100 == 0:
            # if step % 10000000 == 0:
            #     ipdb.set_trace()
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = (
                '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch))

        # write summary every 100 step
        if step % 100 == 0:
            summary_str = sess.run(
                summary_op, feed_dict={is_val: False})
            summary_writer.add_summary(summary_str, step)

        # do validation every 100000 step
        if step % 100000 == 0 or (step + 1) == FLAGS.max_steps:
            np_val_loss = 0
            print('Doing validation, please wait ...')
            for j in range(int(batch_of_val)):
                # images_batch, targets_batch, inf_batch, temp_loss = sess.run(
                #     [images, targets, inf_targets, loss],
                temp_loss, = sess.run(
                    [loss],
                    feed_dict={is_val: True})
                # ipdb.set_trace()
                np_val_loss += temp_loss
            val_audio_rd.dequeue(val_left_to_dequeue)
            mean_val_loss = np_val_loss / batch_of_val
            print('validation loss %.2f' % mean_val_loss)
            val_loss[val_loss_id] = mean_val_loss
            val_loss_id += 1
            np.save('val_loss2.npy', val_loss)

        # store the model every 10000 step
        if step % 10000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


train()
