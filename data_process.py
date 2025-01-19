import os
import numpy as np
import librosa
from scipy.io import wavfile
from scipy import signal
import random
from scipy.fftpack import next_fast_len
from scipy.signal import spectrogram

# 设定随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# 带通滤波器
def band_pass_filter(original_signal, order, fc1, fc2, fs):
    b, a = signal.butter(N=order, Wn=[2*fc1/fs, 2*fc2/fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal

# 预处理音频文件
def process_audio_files():
    for path in ['AS', 'MR', 'MS', 'MVP', 'N']:
        directory = '/root/autodl-tmp/heart-classification/data/' + path
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                audio_path = os.path.join(directory, filename)
                audio_data, fs = librosa.load(audio_path, sr=None)
                audio_data = band_pass_filter(audio_data, 2, 25, 400, fs)
                down_sample_audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=1000)
                down_sample_audio_data = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))
                save_path = os.path.join('/root/autodl-tmp/heart-classification/processed_data1/', path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                wavfile.write(os.path.join(save_path, filename), 1000, down_sample_audio_data)

# Polycoherence functions
def __get_norm(norm):
    if norm == 0 or norm is None:
        return None, None
    else:
        try:
            norm1, norm2 = norm
        except TypeError:
            norm1 = norm2 = norm
        return norm1, norm2

def __freq_ind(freq, f0):
    try:
        return [np.argmin(np.abs(freq - f)) for f in f0]
    except TypeError:
        return np.argmin(np.abs(freq - f0))

def __product_other_freqs(spec, indices, synthetic=(), t=None):
    p1 = np.prod([amplitude * np.exp(2j * np.pi * freq * t + phase)
                  for (freq, amplitude, phase) in synthetic], axis=0)
    p2 = np.prod(spec[:, indices[len(synthetic):]], axis=1)
    return p1 * p2

def _polycoherence_0d(data, fs, *freqs, norm=2, synthetic=(), **kwargs):
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    ind = __freq_ind(freq, freqs)
    sum_ind = __freq_ind(freq, np.sum(freqs))
    spec = np.transpose(spec, [1, 0])
    p1 = __product_other_freqs(spec, ind, synthetic, t)
    p2 = np.conjugate(spec[:, sum_ind])
    coh = np.mean(p1 * p2, axis=0)
    if norm is not None:
        coh = np.abs(coh)
        coh **= 2
        temp2 = np.mean(np.abs(p1) ** norm1 * np.abs(p2) ** norm2, axis=0)
        coh /= temp2
        coh **= 0.5
    return coh

def _polycoherence_1d(data, fs, *freqs, norm=2, synthetic=(), **kwargs):
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    spec = np.transpose(spec, [1, 0])
    ind2 = __freq_ind(freq, freqs)
    ind1 = np.arange(len(freq) - sum(ind2))
    sumind = ind1 + sum(ind2)
    otemp = __product_other_freqs(spec, ind2, synthetic, t)[:, None]
    temp = spec[:, ind1] * otemp
    temp2 = np.mean(np.abs(temp) ** 2, axis=0)
    temp *= np.conjugate(spec[:, sumind])
    coh = np.mean(temp, axis=0)
    if norm is not None:
        coh = np.abs(coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** 2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], coh

def _polycoherence_1d_sum(data, fs, f0, *ofreqs, norm=2, synthetic=(), **kwargs):
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    spec = np.transpose(spec, [1, 0])
    ind3 = __freq_ind(freq, ofreqs)
    otemp = __product_other_freqs(spec, ind3, synthetic, t)[:, None]
    sumind = __freq_ind(freq, f0)
    ind1 = np.arange(np.searchsorted(freq, f0 - np.sum(ofreqs)))
    ind2 = sumind - ind1 - sum(ind3)
    temp = spec[:, ind1] * spec[:, ind2] * otemp
    if norm is not None:
        temp2 = np.mean(np.abs(temp) ** 2, axis=0)
    temp *= np.conjugate(spec[:, sumind, None])
    coh = np.mean(temp, axis=0)
    if norm is not None:
        coh = np.abs(coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** 2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], coh

def _polycoherence_2d(data, fs, *ofreqs, norm=2, flim1=None, flim2=None, synthetic=(), **kwargs):
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode='complex', **kwargs)
    spec = np.require(spec, 'complex64')
    spec = np.transpose(spec, [1, 0])  # transpose (f, t) -> (t, f)
    if flim1 is None:
        flim1 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)
    if flim2 is None:
        flim2 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)
    ind1 = np.arange(*np.searchsorted(freq, flim1))
    ind2 = np.arange(*np.searchsorted(freq, flim2))
    ind3 = __freq_ind(freq, ofreqs)
    otemp = __product_other_freqs(spec, ind3, synthetic, t)[:, None, None]
    sumind = ind1[:, None] + ind2[None, :] + sum(ind3)
    temp = spec[:, ind1, None] * spec[:, None, ind2] * otemp
    if norm is not None:
        temp2 = np.mean(np.abs(temp) ** norm1, axis=0)
    temp *= np.conjugate(spec[:, sumind])
    coh = np.mean(temp, axis=0)
    del temp
    if norm is not None:
        coh = np.abs(coh, out=coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** norm2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], freq[ind2], coh

def polycoherence(data, *args, dim=2, **kwargs):
    N = len(data)
    kwargs.setdefault('nperseg', N // 20)
    kwargs.setdefault('nfft', next_fast_len(N // 10))
    if dim == 0:
        f = _polycoherence_0d
    elif dim == 1:
        f = _polycoherence_1d
    elif dim == 'sum':
        f = _polycoherence_1d_sum
    elif dim == 2:
        f = _polycoherence_2d
    else:
        raise
    return f(data, *args, **kwargs)

# 获取所有文件名
def get_all_filenames(file_dir):
    all_files = [file for file in os.listdir(file_dir)]
    return all_files

# 处理和保存文件
def process_and_save_file(path, name, class_name, dataset_type):
    file_path = os.path.join(path, name)
    sig, sr = librosa.load(file_path, sr=1000)
    _, _, bi_spectrum = polycoherence(sig, nfft=1024, fs=1000, norm=None, noverlap=100, nperseg=256)
    bi_spectrum = np.abs(bi_spectrum)
    bi_spectrum = 255 * (bi_spectrum - np.min(bi_spectrum)) / (np.max(bi_spectrum) - np.min(bi_spectrum))
    bi_spectrum = np.stack((bi_spectrum, bi_spectrum, bi_spectrum), axis=-1)  # Expand to 3 channels

    save_path = os.path.join(f'/root/autodl-tmp/heart-classification/features1/{dataset_type}', class_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, name + '.npy'), bi_spectrum)

# 获取特征并按比例划分数据集
def get_feature(file_folder, class_list, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    for class_name in class_list:
        path = os.path.join(file_folder, class_name)
        all_files = os.listdir(path)
        random.shuffle(all_files)

        total_files = len(all_files)
        train_num = int(total_files * train_ratio)
        val_num = int(total_files * val_ratio)
        test_num = total_files - train_num - val_num

        for name in all_files[:train_num]:
            process_and_save_file(path, name, class_name, 'train')

        for name in all_files[train_num:train_num + val_num]:
            process_and_save_file(path, name, class_name, 'val')

        for name in all_files[train_num + val_num:]:
            process_and_save_file(path, name, class_name, 'test')

# 主函数
if __name__ == "__main__":
    setup_seed(3407)
    process_audio_files()
    file_folder = '/root/autodl-tmp/heart-classification/processed_data1'
    class_name = ['N', 'AS', 'MS', 'MR', 'MVP']
    get_feature(file_folder, class_name)
