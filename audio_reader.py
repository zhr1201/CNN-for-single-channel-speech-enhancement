'''
Class audioreader
    find the speech and noise in the files and enqueue the audios
    that have been read

'''

import tensorflow as tf
import librosa
import threading
import numpy as np
import fnmatch
import os
import random
import ipdb
from numpy.lib import stride_tricks


def find_files(directory, pattern=['*.wav', '*.WAV']):
    '''find files in the directory'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern[0]):
            files.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, pattern[1]):
            files.append(os.path.join(root, filename))
    return files


class Audio_reader(object):
    """reading and framing"""

    def __init__(self,
                 audio_dir,
                 noise_dir,
                 coord,
                 N_IN,
                 frame_length,
                 frame_move,
                 is_val):
        '''coord: tensorflow coordinator
        N_IN: number of input frames presented to DNN
        frame_move: hopsize'''
        self.audio_dir = audio_dir
        self.noise_dir = noise_dir
        self.coord = coord
        self.N_IN = N_IN
        self.frame_length = frame_length
        self.frame_move = frame_move
        self.is_val = is_val
        self.sample_placeholder_many = tf.placeholder(
            tf.float32, shape=(None, self.N_IN, 2, frame_length))
        # queues to store the data
        if not is_val:
            self.q = tf.RandomShuffleQueue(
                200000, 5000, tf.float32, shapes=(self.N_IN, 2, frame_length))
        else:
            self.q = tf.FIFOQueue(
                200000, tf.float32, shapes=(self.N_IN, 2, frame_length))
        self.enqueue_many = self.q.enqueue_many(
            self.sample_placeholder_many + 0)
        self.audiofiles = find_files(audio_dir)
        self.noisefiles = find_files(noise_dir)
        print('%d speech found' % len(self.audiofiles))
        print('%d noise found' % len(self.noisefiles))
        # ipdb.set_trace()

    def dequeue(self, num_elements):
        '''dequeue many element at once'''
        output = self.q.dequeue_many(num_elements)
        return output

    def norm_audio(self):
        '''Normalize the audio files
        used before training using a independent script'''
        for file in self.audiofiles:
            audio, sr = librosa.load(file, sr=16000)
            div_fac = 1 / np.max(np.abs(audio)) / 3.0
            audio = audio * div_fac
            librosa.output.write_wav(file, audio, sr)
        for file in self.noisefiles:
            audio, sr = librosa.load(file, sr=16000)
            div_fac = 1 / np.max(np.abs(audio)) / 3.0
            audio = audio * div_fac
            librosa.output.write_wav(file, audio, sr)

    def thread_main(self, sess):
        '''thread for reading files and enqueue the original
        signal'''
        stop = False
        SNR = [0.0, 0.1, 0.4]  # possible multiply fac adding the signals
        # SNR = [0]
        N_epoch = 1
        N_audio_files = len(self.audiofiles)
        N_noise_files = len(self.noisefiles)
        # total posible combinations
        N_tot = N_noise_files * N_audio_files
        # index: noise audio N_snr
        count = 0
        while not stop:
            # randomly comnbine the speech and noise
            ids = range(N_tot)
            random.shuffle(ids)
            for i in ids:
                # ipdb.set_trace()
                noise_id = i / (N_audio_files)
                audio_id = i - N_audio_files * noise_id
                audio_org, _ = librosa.load(self.audiofiles[audio_id], sr=None)
                noise_org, _ = librosa.load(self.noisefiles[noise_id], sr=None)
                audio_len = len(audio_org)
                noise_len = len(noise_org)
                # print('%d %d' % (audio_len, noise_len))
                # trim the signals into same length and add
                tot_len = max(audio_len, noise_len)
                if audio_len < noise_len:
                    rep_time = int(np.floor(noise_len / audio_len))
                    left_len = noise_len - audio_len * rep_time
                    temp_data = np.tile(audio_org, [1, rep_time])
                    temp_data.shape = (temp_data.shape[1], )
                    audio = np.hstack((
                        temp_data, audio_org[:left_len]))
                    noise = np.array(noise_org)
                else:
                    rep_time = int(np.floor(audio_len / noise_len))
                    left_len = audio_len - noise_len * rep_time
                    temp_data = np.tile(noise_org, [1, rep_time])
                    temp_data.shape = (temp_data.shape[1], )
                    noise = np.hstack((
                        temp_data, noise_org[:left_len]))
                    audio = np.array(audio_org)

                # number of generated frames
                num_iter = np.floor(
                    (tot_len - self.frame_length) / self.frame_move -
                    self.N_IN)
                # generate for each multiply factor
                for mul_fac in SNR:
                    noisy_speech = audio + mul_fac * noise
                    noisy_speech.shape = (1, -1)
                    audio.shape = (1, -1)
                    # ipdb.set_trace()
                    data = np.concatenate((noisy_speech, audio))
                    data_frames = stride_tricks.as_strided(
                        data,
                        shape=(num_iter, self.N_IN, 2, self.frame_length),
                        strides=(
                            data.strides[1] * self.frame_move,
                            data.strides[1] * self.frame_move,
                            data.strides[0],
                            data.strides[1]))
                    # enqueue the signals
                    sess.run(
                        self.enqueue_many,
                        feed_dict={self.sample_placeholder_many:
                                   data_frames})
                    count += num_iter
                if not self.is_val and i % 100 == 0:
                    print('epoch %d' % N_epoch)
            if not self.is_val:
                print('end of an epoch with %d samples'
                      % count)
            np.save('sampleN.npy', count)

    def start_threads(self, sess, num_thread=1):
        '''start the threads'''
        for i in range(num_thread):
            thread = threading.Thread(
                target=self.thread_main, args=(sess, ))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
