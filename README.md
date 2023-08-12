
# Audio Deepfake Detection using Spectrograms

This python based model built using TensorFlow is predominantly made for the classification of synthetic audio. In this model,  we will feed Mel-Spectrogram feature to our model as input. We can simply consider Mel-Spectrogram as 2D image which x-axis represents time and y-axis represents frequency information. Hence, we can consider this problem as an Image Classification Problem. 


## Problem Statement

Audio Deepfakes, technically known as logical-access voice spoofing techniques, have become an increased threat on voice interfaces due to the recent breakthroughs in speech synthesis and voice conversion technologies. As new types of speech synthesis and voice conversion techniques are emerging rapidly, the generalization ability of spoofing countermeasures is becoming an increasingly critical challenge. To detect these counterfeit audios from being misused at a small scale or at a national level, we must formulate a program which can confirm the authenticity of it.


## Methodology

First, Spectrogram feature is extracted from audio sample using Short Time Fourier Transform (STFT) which is a variant of Fast Fourier Transform (FFT). The fast Fourier transform is a powerful tool that allows us to analyze the frequency content of a signal. For non-periodic signals such as Music and Speech, compute several spectrums by performing FFT on several windowed segments of the signal hence it is called Short-time Fourier transform. It is computed on overlapping windowed segments of the signal, and we get what is called the spectrogram. STFT suffers from famous Time-Frequency localization trade-off, which is:

* Narrow-Window results in Good time resolution but Poor
          frequency resolution
* Wide-Window results in Poor time resolution but Good
          frequency resolution

Second, Mel-Spectrogram is computed which is a Spectrogram in Mel-scale (logarithmic transformation of a signal's frequency). Mel spectrograms are better suited for applications that need to model human hearing perception such as Speech and Music.

## Motivation

Computer Generated voices which are identical to a targeted person's voice, this feature has been widely exploited and used in unethical ways to  manipulate people for gaining private information with advancement in technology nowadays it has become really difficult to differentiate between  real and fake audios and currently in the market there are various ways to generate fake audios like voice conversion system text-to-speech system and  replay attacks therefore a system which can accurately differentiate between real and fake audios is globally demanded.

## About the Model

This notebook demonstrates Fake Speech Detection with TensorFlow. It shows how to use tf.data, tfrecord for Audio Processing task and also demonstrates how to extract Spectrogram features from Raw Audio using TensorFlow. This notebook will also show how to set up Augmentation Pipeline for audio data. It will also implement some cool augmentations such as CutMix and MixUp for Audio data. It also shows how we can use Automatic Speech Recognition (ASR) model for Audio Classification task to take leverage of Relevant Transfer Learning and avoid ImageNet pretraining. n this project, I'll be showing the usage of Conformer: Convolution-augmented Transformer for Speech Recognition Model by Google using Tensorflow. We also see how is TensorFlow audio_classification_models library used to load similar models with just one line of code for a similar task.


ASVspoof2019 dataset has been chosen for Fake Speech Detection task. This dataset comes with dev/valid & eval/test data. Hence we can directly use them for comparing model's performance. TFRecord dataset for this Audio Processing task is created using ASVspoof 2019 tfrecord Data notebook. TFRecord files are created using StratifiedFold to stratify each .tfrec file for avoiding imbalance in batch during training. This notebook is compatible with both GPU and TPU. Device is automatically selected so you won't have to do anything to allocate device.

## Dataset

The dataset used was ASVspoof 2019 dataset and tfrecord dataset. The links for the same have been provided below:

ASVspoof 2019 dataset   
https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

ASVspoof 2019 tfrecord dataset  
https://www.kaggle.com/datasets/awsaf49/asvspoof-2019-tfrecord-dataset

## Author

Surbhit Pratik (https://github.com/surbhit20)
