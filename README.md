# CNN-for-single-channel-speech-enhancement
A tensorflow implementation of the paper: A Fully Convolutional Neural Network for Speech Enhancement
https://arxiv.org/abs/1609.07132
A processed sample can be found in audiosample/

## Requirements
  * tensorflow r0.11
  * librosa
  * numpy

## File documentation
  * SENN.py: The structure of the network.
  * audio_reader.py: Find the speech and noise in the files and enqueue the audios that have been read into tf.queue.
  * SENN_train.py: Train the net.
  * SENN_audio_eval: Use a noisy sample to evaluate the net/

## Training procedure
  * Orgnize your clean speech files and noise files in different directories.
  * Change their dir in SENN_train.py and train the net.
  * Mix your own samples and test use SENN_audio_eval.py
  
## Some other things
  The original paper use  per sample pre-whitening and we also use that in this piece of code, but it turns out better to use global
  mean and var to do the pre-whitening.
  We didn't use the skip connections and our tests show that the most important factor leading to good performance is the size of the noise data set. The model is very likely to overfit if only 100 types of noise are provided.
