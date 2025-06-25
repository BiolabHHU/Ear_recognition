import glob
import os
import numpy as np
import resampy
import torch
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import MFCC

EPS = 1e-20

class Dataset(Dataset):
    def __init__(self,
                 sampling_rate=16000,
                 mfcc_numcep=13,
                 mfcc_nfilt=26,
                 mfcc_nfft=512,
                 mfcc_winlen=0.016,
                 mfcc_winstep=0.01,
                 data_folder_path='/home/hhdx/PycharmProjects/yyds/yds_paper/ear_recognition'):
        self.samples = []
        self.labels = []
        self.n_samples = 0
        self.unique_labels = []
        self.train_samples = []
        self.train_labels = []
        self.val_samples = []
        self.val_labels = []
        self.test_samples = []
        self.test_labels = []

        self.samples1 = []
        self.labels1 = []
        self.unique_labels1 = []
        self.train_samples1 = []
        # self.train_labels1 = []
        self.val_samples1 = []
        # self.val_labels1 = []
        self.test_samples1 = []
        self.test_labels1 = []
        self.data_folder_path = data_folder_path
        self.sampling_rate = sampling_rate
        self.mfcc_numcep = mfcc_numcep
        self.mfcc_nfilt = mfcc_nfilt
        self.mfcc_nfft = mfcc_nfft
        self.mfcc_winlen = mfcc_winlen
        self.mfcc_winstep = mfcc_winstep

    def init_samples_and_labels(self):
        """
        This method initalizes the dataset by collectiong all available train and test data samples.
        The train samples are randomly split into 90% train and 10% validation.
        Even though all samples are collected only the currently active ones will be returned with get.
        To set which samples are currently active call load_data(self, train=False, val=False, test=False)
        and set the wanted samples to true
        """
        vox_train_path = self.data_folder_path + '/data_100/train_ear60/*/*.wav'
        vox_test_path = self.data_folder_path + '/data_100/train_ear30/*/*.wav'

        # Get the paths to all train and val data samples
        globs = glob.glob(vox_train_path)
        print('collectiong training and validation samples')

        # Gat the list of samples and labels
        samples = [(sample, 'none') for sample in globs]  # 将每个训练样本的文件路径与标签（'none'）组成元组，并存储在列表中。
        labels = [os.path.basename(os.path.dirname(f)) for f in globs]  # 从每个训练样本的文件路径中提取标签，并存储在列表中。

        unique_labels = np.unique(labels)
        print('found:')
        print(len(unique_labels), ' unique speakers')
        print(len(samples), ' total voice samples including augmentations')
        print('splitting into 90% training and 10% validation')

        skf = StratifiedKFold(n_splits=10, shuffle=True)  # 使用StratifiedKFold方法创建一个分层k折交叉验证对象，用于将训练数据集划分为训练集和验证集。
        train_index, val_index = [], []
        for traini, vali in skf.split(samples, labels):
            if (len(vali) == int(round(len(samples) / 10))):
                train_index = traini
                val_index = vali
        if (len(train_index) <= 1):
            print('StratifiedKFold Failed')

        self.train_samples = list(np.array(samples)[train_index])
        self.train_labels = list(np.array(labels)[train_index])
        self.val_samples = list(np.array(samples)[val_index])
        self.val_labels = list(np.array(labels)[val_index])

        # Get the paths to all test data samples
        globs = glob.glob(vox_test_path)
        print('collectiong test samples')

        # Gat the list of samples and labels
        test_samples = [(sample, 'none') for sample in globs]
        test_labels = [os.path.basename(os.path.dirname(f)) for f in globs]

        unique_labels = np.unique(test_labels)
        print('found:')
        print(len(unique_labels), ' unique speakers')
        print(len(test_samples), ' voice samples')
        print('DONE collectiong samples')

        self.test_samples = list(np.array(test_samples))
        self.test_labels = list(np.array(test_labels))

        # #####################################################
        # 添加声纹数据
        '''
        vox_train_path_1 = self.data_folder_path + '/data_100/train_voice60/*/*.wav'
        vox_test_path_1 = self.data_folder_path + '/data_100/train_voice30/*/*.wav'

        globs1 = glob.glob(vox_train_path_1)
        samples1 = [(sample, 'none') for sample in globs1]
        labels1 = [os.path.basename(os.path.dirname(f)) for f in globs1]

        skf = StratifiedKFold(n_splits=10, shuffle=True)  # 使用StratifiedKFold方法创建一个分层k折交叉验证对象，用于将训练数据集划分为训练集和验证集。
        train_index, val_index = [], []
        for traini, vali in skf.split(samples1, labels1):
            if (len(vali) == int(round(len(samples1) / 10))):
                train_index = traini
                val_index = vali
        if (len(train_index) <= 1):
            print('StratifiedKFold Failed')

        self.train_samples1 = list(np.array(samples1)[train_index])
        self.val_samples1 = list(np.array(samples1)[val_index])

        globs = glob.glob(vox_test_path_1)
        test_samples1 = [(sample, 'none') for sample in globs]
        test_labels1 = [os.path.basename(os.path.dirname(f)) for f in globs]

        self.test_samples1 = list(np.array(test_samples1))
        self.test_labels1 = list(np.array(test_labels1))
        '''
    def __getitem__(self, index):

        sample_path, augmentation = self.samples[index]
        # 移除空字符
        rate, sample = wavfile.read(sample_path, np.dtype)
        sample = sample[:, 1]
        sample = resampy.resample(sample, rate, self.sampling_rate)

        # 计算MFCC特征
        augmented_sample = MFCC.mfcc(sample, self.sampling_rate, numcep=self.mfcc_numcep, nfilt=self.mfcc_nfilt,
                                nfft=self.mfcc_nfft, winlen=self.mfcc_winlen, winstep=self.mfcc_winstep)
        # augmented_sample = augmented_sample[:, :13]  # 提取前13维
        '''
        if augmented_sample.shape[0] < 99:
            num_rows_to_add = 99 - augmented_sample.shape[0]
            padding_rows = np.zeros((num_rows_to_add, augmented_sample.shape[1]))
            augmented_sample = np.concatenate((augmented_sample, padding_rows), axis=0)
        augmented_sample = augmented_sample[0:99, :]
        '''
        # 计算加入声纹数据MFCC
        '''
        sample_voice_path, augmentation = self.samples1[index]
        rate1, sample_voice = wavfile.read(sample_voice_path, np.dtype)
        sample_voice = resampy.resample(sample_voice, rate1, self.sampling_rate)
        augmented_sample_voice = mfcc(sample_voice, self.sampling_rate, numcep=self.mfcc_numcep, nfilt=self.mfcc_nfilt,
                                      nfft=self.mfcc_nfft, winlen=self.mfcc_winlen, winstep=self.mfcc_winstep)
        # augmented_sample_voice = augmented_sample_voice[:, :13]  # 提取最后10维
        if augmented_sample_voice.shape[0] < 99:
            num_rows_to_add = 99 - augmented_sample_voice.shape[0]
            padding_rows = np.zeros((num_rows_to_add, augmented_sample_voice.shape[1]))
            augmented_sample_voice = np.concatenate((augmented_sample_voice, padding_rows), axis=0)

        augmented_sample_voice1 = augmented_sample_voice[0:99, :]
        augmented_sample_add = np.concatenate((augmented_sample, augmented_sample_voice1), axis=0)
        '''
        label = self.unique_labels.index(self.labels[index])
        id = '/'.join(sample_path.rsplit('\\')[-2:])
        return torch.from_numpy(augmented_sample), label, id

    def __len__(self):

        return self.n_samples

    def load_data(self, train=False, val=False, test=False):

        self.samples = []
        self.labels = []
        self.n_samples = []
        self.unique_labels = []
        # self.samples1 = []
        # self.labels1 = []
        # self.unique_labels1 = []

        if (train):
            self.samples = self.samples + self.train_samples
            self.labels = self.labels + self.train_labels

            # self.samples1 = self.samples1 + self.train_samples1

        if (val):
            self.samples = self.samples + self.val_samples
            self.labels = self.labels + self.val_labels

            # self.samples1 = self.samples1 + self.val_samples1

        if (test):
            self.samples = self.samples + self.test_samples
            self.labels = self.labels + self.test_labels

            # self.samples1 = self.samples1 + self.test_samples1

        # Get the num of samples and the unique class names
        self.n_samples = len(self.samples)
        self.unique_labels = list(np.unique(self.labels))

'''
# 创建 Dataset 类的实例
dataset = Dataset()

# 调用 init_samples_and_labels 方法
dataset.init_samples_and_labels()

# 调用 load_data 方法，加载需要的数据
dataset.load_data(train=True, val=True, test=True)

# 调用 __getitem__ 方法获取指定索引的数据
index = 0  # 要获取的样本索引
sample, label, id = dataset[index]

# 调用 __len__ 方法获取样本数量
n_samples = len(dataset)
'''
