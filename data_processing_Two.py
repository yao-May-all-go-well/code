import os
import numpy as np
import librosa
from scipy.io import wavfile
from scipy import signal
import random
from extract_bispectrum import polycoherence


def setup_seed(seed):
    # Python随机种子
    random.seed(seed)
    # NumPy随机种子
    np.random.seed(seed)
    # PyTorch随机种子
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # 禁用CuDNN以确保确定性


def band_pass_filter(original_signal, order, fc1, fc2, fs):
    b, a = signal.butter(N=order, Wn=[2 * fc1 / fs, 2 * fc2 / fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal


def process_audio_files():
    base_directory = f'/root/autodl-tmp/heart-classification/trains/'
    save_directory = f'/root/autodl-tmp/heart-classification/processed_data_2/'

    for class_name in ['abnormal', 'abnormal_syn_roll', 'abnormal_synthetic', 'normal']:
        directory = os.path.join(base_directory, class_name)
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                audio_path = os.path.join(directory, filename)
                audio_data, fs = librosa.load(audio_path, sr=None)
                audio_data = band_pass_filter(audio_data, 2, 25, 400, fs)
                down_sample_audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=1000)
                down_sample_audio_data = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))

                save_path = os.path.join(save_directory, class_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                wavfile.write(os.path.join(save_path, filename), 1000, down_sample_audio_data)


def extract_features(dataset_type):
    base_directory = f'/root/autodl-tmp/heart-classification/processed_data_2/'

    for class_name in ['abnormal', 'abnormal_syn_roll', 'abnormal_synthetic', 'normal']:
        directory = os.path.join(base_directory, class_name)
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                file_path = os.path.join(directory, filename)
                sig, sr = librosa.load(file_path, sr=1000)
                _, _, bi_spectrum = polycoherence(sig, nfft=1024, fs=1000, norm=None, noverlap=100, nperseg=256)
                bi_spectrum = np.abs(bi_spectrum)
                bi_spectrum = 255 * (bi_spectrum - np.min(bi_spectrum)) / (np.max(bi_spectrum) - np.min(bi_spectrum))
                bi_spectrum = bi_spectrum.reshape((1, 256, 256))

                save_path = os.path.join(f'/root/autodl-tmp/heart-classification/features_Two/',class_name)
                np.save(os.path.join(save_path, filename.replace('.wav', '.npy')), bi_spectrum)


def process_dataset():
    process_audio_files()
    extract_features()

if __name__ == "__main__":
    setup_seed(3407)
    process_dataset()

