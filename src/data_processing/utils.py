
import os
import torch
import torchaudio
from torchaudio import transforms as T
from scipy.signal import butter, lfilter
import pandas as pd
import librosa
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import time
import pdb


def random_multiply(data):
    new_data = data.copy()
    return new_data * (0.9 + random.random() / 5.)


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def _slice_data_librosa(start, end, data, sample_rate):
    """
    RespireNet paper..
    sample_rate denotes how many sample points for one second
    """
    max_ind = len(data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)

    return data[start_ind: end_ind]


def get_entire_signal_librosa(data, input_sec=8, sample_rate=16000, butterworth_filter=None, spectrogram=False, pad=False, from_cycle=False, yt=None, pad_type='repeat'):

    if butterworth_filter:
        # butter bandpass filter
        data = _butter_bandpass_filter(lowcut=200, highcut=1800, fs=sample_rate, order=butterworth_filter)

    # Trim leading and trailing silence from an audio signal.
    FRAME_LEN = int(sample_rate / 10)  # 
    HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length

    TRIM = True
    if TRIM:
        yt, index = librosa.effects.trim(
                    data, frame_length=FRAME_LEN, hop_length=HOP
                )
    else:
        yt = data
        
    # check audio not too short
    duration = librosa.get_duration(y=yt, sr=sample_rate)
    # pdb.set_trace()
    if duration < input_sec:
        if not pad:
            print("Warning: audio too short, skipped")
            return None
        else:
            # yt = split_pad_sample([yt, 0,0], input_sec, sample_rate, pad_type)[0][0]
            yt = split_pad_sample(yt, input_sec, sample_rate, pad_type)[0]
        # pdb.set_trace()
    
    # directly process to spectrogram
    if spectrogram:
        return pre_process_audio_mel_t(yt.squeeze(), f_max=8000)

    return yt


def get_split_signal_librosa(data, input_sec=8, sample_rate=16000, butterworth_filter=None, spectrogram=False, trim_tail=False, pad_type="zero"):
    rate = sample_rate

    if butterworth_filter:
        # butter bandpass filter
        data = _butter_bandpass_filter(
            lowcut=200, highcut=1800, fs=sample_rate, order=butterworth_filter)

    # Trim leading and trailing silence from an audio signal.
    FRAME_LEN = int(sample_rate / 10)  #
    HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
    
    yt, index = librosa.effects.trim(
        data, frame_length=FRAME_LEN, hop_length=HOP
    )

    drop_last = False
    if trim_tail:
        drop_last = decide_droplast(yt, rate, input_sec)

    # audio_chunks = [res[0] for res in split_pad_sample([yt, 0, 0], input_sec, rate, types=pad_type)]
    audio_chunks = [res for res in split_pad_sample(yt, input_sec, rate, types=pad_type)]
    
    if drop_last:
        audio_chunks.pop()

    if not spectrogram:
        return audio_chunks

    # directly process to spectrogram
    audio_image = []
    for audio in audio_chunks:
        image = pre_process_audio_mel_t(audio.squeeze(), f_max=8000)
        audio_image.append(image)
    return audio_image


def get_entire_signal_fbank(data, sample_rate=16000):
    # Trim leading and trailing silence from an audio signal.
    FRAME_LEN = int(sample_rate / 10)  # 
    HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
    yt, index = librosa.effects.trim(
                data, frame_length=FRAME_LEN, hop_length=HOP
            )

    waveform = yt
    waveform = waveform - waveform.mean()
    waveform = torch.tensor(waveform).reshape([1,-1])
    fbank = torchaudio.compliance.kaldi.fbank(waveform, channel=0, frame_length=25, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
    return fbank


def get_split_signal_fbank(data, input_sec=10, sample_rate=16000):

    # Trim leading and trailing silence from an audio signal.
    FRAME_LEN = int(sample_rate / 10)  # 
    HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
    yt, index = librosa.effects.trim(
                data, frame_length=FRAME_LEN, hop_length=HOP
            )

    audio_chunks = [res for res in split_pad_sample(yt, input_sec, sample_rate, types="zero")]
    
    
    # directly process to spectrogram
    audio_image = []
    for waveform in audio_chunks:
        waveform = waveform - waveform.mean()
        waveform = torch.tensor(waveform).reshape([1,-1])
        #print(waveform.shape)
        if waveform.shape[1] > 400: 
            fbank = torchaudio.compliance.kaldi.fbank(waveform, channel=0, frame_length=25, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        

            #print( waveform.shape[1]/sample_rate, fbank.shape)
            audio_image.append(fbank)
    return audio_image


def pre_process_audio_mel_t(audio, sample_rate=16000, n_mels=64, f_min=50, f_max=2000, nfft=1024, hop=512):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    # convert scale to dB from magnitude
    S = librosa.power_to_db(S, ref=np.max)
    if S.max() != S.min():
        mel_db = (S - S.min()) / (S.max() - S.min())
    else:
        mel_db = S
        print("warning in producing spectrogram!")

    return mel_db.T


def _zero_padding(source, output_length):
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)

    frac = src_length / output_length
    if frac < 0.5:
        # tile forward sounds to fill empty space
        cursor = 0
        while (cursor + src_length) < output_length:
            try:
                copy[cursor:(cursor + src_length)] = source[:]
            except:
                pdb.set_trace()
            # pdb.set_trace()
            cursor += src_length
    else:
        # [src_length:] part will be zeros
        copy[:src_length] = source[:]

    return copy


def _equally_slice_pad_sample(sample, desired_length, sample_rate):
    """
    pad_type == 0: zero-padding
    if sample length > desired_length, 
    all equally sliced samples with samples_per_slice number are zero-padded or recursively duplicated
    """
    output_length = int(
        desired_length * sample_rate)  # desired_length is second
    soundclip = sample.copy()
    # pdb.set_trace()
    n_samples = len(soundclip)

    total_length = n_samples / sample_rate  # length of cycle in seconds
    # get the minimum number of slices needed
    n_slices = int(math.ceil(total_length / desired_length))
    samples_per_slice = n_samples // n_slices
    # pdb.set_trace()
    output = []  # holds the resultant slices
    src_start = 0  # staring index of the samples to copy from the sample buffer
    for i in range(n_slices):
        src_end = min(src_start + samples_per_slice, n_samples)
        length = src_end - src_start

        copy = _zero_padding(soundclip[src_start:src_end], output_length)
        # output.append((copy, sample[1], sample[2]))
        output.append(copy)
        src_start += length

    return output


def _duplicate_padding(sample, source, output_length, sample_rate, types):
    # pad_type == 1 or 2
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    left = output_length - src_length  # amount to be padded

    if types == 'repeat':
        aug = sample
    # else:
    #     aug = augment_raw_audio(sample, sample_rate)

    while len(aug) < left:
        aug = np.concatenate([aug, aug])

    random.seed(7456)
    prob = random.random()
    if prob < 0.5:
        # pad the back part of original sample
        copy[left:] = source
        copy[:left] = aug[len(aug)-left:]
    else:
        # pad the front part of original sample
        copy[:src_length] = source[:]
        copy[src_length:] = aug[:left]

    return copy


def split_pad_sample(sample, desired_length, sample_rate, types='repeat'):
    """
    if the audio sample length > desired_length, then split and pad samples
    else simply pad samples according to pad_types
    * types 'zero'   : simply pad by zeros (zero-padding)
    * types 'repeat' : pad with duplicate on both sides (half-n-half)
    * types 'aug'    : pad with augmented sample on both sides (half-n-half)	
    """
    if types == 'zero':
        return _equally_slice_pad_sample(sample, desired_length, sample_rate)

    output_length = int(desired_length * sample_rate)
    soundclip = sample[0].copy()
    n_samples = len(soundclip)

    output = []
    if n_samples > output_length:
        """
        if sample length > desired_length, slice samples with desired_length then just use them,
        and the last sample is padded according to the padding types
        """
        # frames[j] = x[j * hop_length : j * hop_length + frame_length]
        frames = librosa.util.frame(
            soundclip, frame_length=output_length, hop_length=output_length//2, axis=0)
        for i in range(frames.shape[0]):
            output.append((frames[i], sample[1], sample[2]))

        # get the last sample
        last_id = frames.shape[0] * (output_length//2)
        last_sample = soundclip[last_id:]

        padded = _duplicate_padding(
            soundclip, last_sample, output_length, sample_rate, types)
        output.append((padded, sample[1], sample[2]))
    else:  # only pad
        padded = _duplicate_padding(
            soundclip, soundclip, output_length, sample_rate, types)
        output.append((padded, sample[1], sample[2]))

    return output


