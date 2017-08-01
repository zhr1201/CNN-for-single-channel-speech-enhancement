'''
Class SENN: speech enhancement neural network
    1. transforming the original signal frames into
        the features fed to the net.
    2. defination of the tensorflow computational graph
        that enhance the speech.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import os
import ipdb
# import sys
import tensorflow as tf
import numpy as np


log10_fac = 1 / np.log(10)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor
    (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        tensor_name = var.op.name
        mean = tf.reduce_mean(var)
        tf.scalar_summary(tensor_name + 'mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary(tensor_name + 'stddev', stddev)
        tf.scalar_summary(tensor_name + 'max', tf.reduce_max(var))
        tf.scalar_summary(tensor_name + 'min', tf.reduce_min(var))
        tf.histogram_summary(tensor_name + 'histogram', var)


def conv2d(x, W):
    '''1 dimentional convolution difined in the paper
    the function's name is not appropriate and
    we didn't change that'''
    return tf.nn.conv2d(x, W, strides=[1, 100, 1, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class SE_NET(object):
    """Class:speech enhancement net"""
    def __init__(self, batch_size, NEFF, N_IN, N_OUT, DECAY=0.999):
        '''NEFF: number of effective FFT points
        N_IN: number of input frames into the nets
        N_OUT: only tested for 1, errors may occur for other number
        DECAY: decay for global mean and var estimation using batch norm
        '''
        self.batch_size = batch_size
        self.NEFF = NEFF
        self.N_IN = N_IN
        self.N_OUT = N_OUT
        self.DECAY = DECAY

    def inputs(self, raw_data_batch):
        '''transform the raw data_batch into
        the input for the nets
        it runs really fast and we don't need to store
        all the mixed samples'''
        # ipdb.set_trace()
        # transpose for FFT
        # shape:
        # batch, N_IN, 2, frame_length to 2 batch N_in frame_length
        raw_data_batch_t = tf.transpose(raw_data_batch, [2, 0, 1, 3])
        raw_data = raw_data_batch_t[0][:][:][:]
        raw_speech = raw_data_batch_t[1][:][:][:]

        # FFT
        # shape:
        # batch, N_in, NFFT
        data_f0 = tf.fft(tf.cast(raw_data, tf.complex64))
        # shape:
        # NFFT, batch, N_in
        data_f1 = tf.transpose(data_f0, [2, 0, 1])
        data_f2 = data_f1[0:self.NEFF][:][:]
        # shape:
        # batch, N_in, NEFF
        data_f3 = tf.transpose(data_f2, [1, 2, 0])
        data_f4 = tf.square(tf.real(data_f3)) + tf.square(tf.imag(data_f3))
        # limiting the minimum value
        data_f5 = tf.maximum(data_f4, 1e-10)
        # into log spectrum
        data_f = 10 * tf.log(data_f5 * 10000) * log10_fac
        # same operational for reference speech
        speech_f0 = tf.fft(tf.cast(raw_speech, tf.complex64))
        speech_f1 = tf.transpose(speech_f0, [2, 0, 1])
        speech_f2 = speech_f1[0:self.NEFF][:][:]
        speech_f3 = tf.transpose(speech_f2, [1, 2, 0])
        speech_f4 = tf.square(
            tf.real(speech_f3)) + tf.square(tf.imag(speech_f3))
        speech_f5 = tf.maximum(speech_f4, 1e-10)
        speech_f = 10 * tf.log(speech_f5 * 10000) * log10_fac

        # shape:
        # batch, N_in, NEFF
        images = data_f
        targets = tf.concat(
            0,
            [tf.reshape(
                speech_f[i][self.N_IN - 1][0:self.NEFF],
                [1, self.NEFF])
             for i in range(0, self.batch_size, 1)])
        # do per image whitening (not batch normalization!)
        images_reshape = tf.transpose(tf.reshape(
            images, [self.batch_size, -1]))
        targets_reshape = tf.transpose(tf.reshape(
            targets, [self.batch_size, -1]))
        batch_mean, batch_var = tf.nn.moments(images_reshape, [0])
        images_reshape_norm = tf.nn.batch_normalization(
            images_reshape, batch_mean, batch_var, 0, 1, 1e-10)
        targets_reshape_norm = tf.nn.batch_normalization(
            targets_reshape, batch_mean, batch_var, 0, 1, 1e-10)
        # ipdb.set_trace()
        images_norm = tf.reshape(tf.transpose(images_reshape_norm),
                                 [self.batch_size, self.N_IN, self.NEFF])
        targets_norm = tf.reshape(tf.transpose(targets_reshape_norm),
                                  [self.batch_size, self.NEFF])
        return images_norm, targets_norm

    def _batch_norm_wrapper(self, inputs, is_trianing, epsilon=1e-6):
        '''wrap up all the operations needed for batch norm
        is_training: true -> using batch property
                     false -> using global(population) property'''
        decay = self.DECAY
        scale = tf.Variable(tf.ones(inputs.get_shape()[-1]))
        beta = tf.Variable(tf.zeros(inputs.get_shape()[-1]))

        # population mean and var
        pop_mean = tf.Variable(
            tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(
            tf.ones([inputs.get_shape()[-1]]), trainable=False)
        if is_trianing:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            # update estimation
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay +
                                   batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay +
                                  batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(
                    inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(
                inputs, pop_mean, pop_var, beta, scale, epsilon)

    def _conv_layer_wrapper(self,
                            input, out_feature_maps, filter_length, is_train):
        '''wrap up all the ops for convolution'''
        filter_width = input.get_shape()[1].value
        in_feature_maps = input.get_shape()[-1].value
        W_conv = weight_variable(
            [filter_width, filter_length, in_feature_maps, out_feature_maps])
        b_conv = bias_variable([out_feature_maps])
        h_conv_t = conv2d(input, W_conv) + b_conv
        # use batch norm
        h_conv_b = self._batch_norm_wrapper(h_conv_t, is_train)
        return tf.nn.relu(h_conv_b)

    def inference(self, images, is_train):
        '''Net configuration as the original paper'''
        image_input = tf.reshape(images, [-1, self.N_IN, self.NEFF, 1])
        # ipdb.set_trace()
        with tf.variable_scope('con1') as scope:
            h_conv1 = self._conv_layer_wrapper(image_input, 12, 13, is_train)
        with tf.variable_scope('con2') as scope:
            h_conv2 = self._conv_layer_wrapper(h_conv1, 16, 11, is_train)
        with tf.variable_scope('con3') as scope:
            h_conv3 = self._conv_layer_wrapper(h_conv2, 20, 9, is_train)
        with tf.variable_scope('con4') as scope:
            h_conv4 = self._conv_layer_wrapper(h_conv3, 24, 7, is_train)
        with tf.variable_scope('con5') as scope:
            h_conv5 = self._conv_layer_wrapper(h_conv4, 32, 7, is_train)
        with tf.variable_scope('con6') as scope:
            h_conv6 = self._conv_layer_wrapper(h_conv5, 24, 7, is_train)
        with tf.variable_scope('con7') as scope:
            h_conv7 = self._conv_layer_wrapper(h_conv6, 20, 9, is_train)
        with tf.variable_scope('con8') as scope:
            h_conv8 = self._conv_layer_wrapper(h_conv7, 16, 11, is_train)
        with tf.variable_scope('con9') as scope:
            h_conv9 = self._conv_layer_wrapper(h_conv8, 12, 13, is_train)
        with tf.variable_scope('con10') as scope:
            f_w = h_conv9.get_shape()[1].value
            i_fm = h_conv9.get_shape()[-1].value
            W_con10 = weight_variable(
                [f_w, 129, i_fm, 1])
            b_conv10 = bias_variable([1])
            h_conv10 = conv2d(h_conv9, W_con10) + b_conv10
        return tf.reshape(h_conv10, [-1, self.NEFF])

    def loss(self, inf_targets, targets):
        '''l2 loss for the log spectrum'''
        loss_v = tf.nn.l2_loss(inf_targets - targets) / self.batch_size
        tf.scalar_summary('loss', loss_v)
        # loss_merge = tf.cond(
        #     is_val, lambda: tf.scalar_summary('val_loss_batch', loss_v),
        #     lambda: tf.scalar_summary('loss', loss_v))
        return loss_v
        # return tf.reduce_mean(tf.nn.l2_loss(inf_targets - targets))

    def train(self, loss, lr):
        '''optimizer'''
        # optimizer = tf.train.GradientDescentOptimizer(0.01)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        train_op = optimizer.minimize(loss)
        return train_op
