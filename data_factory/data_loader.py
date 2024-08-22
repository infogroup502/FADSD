import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        print(f"Index before conversion: {index}")
        index = index * self.step
        print(f"Index before conversion: {index}")
        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])

class GenesisSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        #print(f"Index before conversion: {index}")
        index = index * self.step
        #print(f"Index before conversion: {index}")
        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])

# class GenesisSegLoader(object):
#     def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
#         self.mode = mode
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#
#         # 加载并缩放训练数据
#         data = np.load(data_path + "/Genesis_train.npy")
#         data = data.reshape(-1, 1)  # 确保数据为二维 (样本数, 特征数)
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#
#         # 加载并缩放测试数据
#         test_data = np.load(data_path + "/Genesis_test.npy")
#         test_data = test_data.reshape(-1, 1)  # 确保数据为二维
#         self.test = self.scaler.transform(test_data)
#
#         self.train = data
#         self.val = self.test
#         self.test_labels = np.load(data_path + "/Genesis_test_label.npy")
#         self.win_size_1 = win_size_1
#         self.count = count
#
#         if self.mode == "val":
#             print("train:", self.train.shape)
#             print("test:", self.test.shape)
#
#     def __len__(self):
#         if self.mode == "train":
#             return self.train.shape[0]
#         elif self.mode == "val":
#             return self.val.shape[0]
#         elif self.mode == "test":
#             return self.test.shape[0]
#         else:
#             return self.test.shape[0]
#
#     def __getitem__(self, index):
#         count = self.count
#         index = index * self.step
#
#         # 第一个数据块索引和提取
#         index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
#         index_1 = np.clip(index_1, 0, len(self.test) - 1)
#         data_block_1 = self.test[index_1]
#
#         # 第二个数据块索引和提取
#         index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
#                             index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
#         index_2 = np.clip(index_2, 0, len(self.test) - 1)
#         data_block_2 = self.test[index_2]
#         data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, 1)  # 调整为适合单变量数据
#
#         return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class SKABSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        # print(f"Index before conversion: {index}")
        index = index * self.step
        # print(f"Index before conversion: {index}")

        # 处理错误，例如跳过此索引或设置一个默认值

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


# class DodgersSegLoader(object):
#     def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
#         self.mode = mode
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#
#         # 加载 train 数据
#         data = pd.read_csv(data_path + '/train.csv')
#         data = data.values  # 数据只有一列，获取所有行
#         data = np.nan_to_num(data)
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#
#         # 加载 test 数据
#         test_data = pd.read_csv(data_path + '/test.csv')
#         test_data = test_data.values  # 数据只有一列，获取所有行
#         test_data = np.nan_to_num(test_data)
#         self.test = self.scaler.transform(test_data)
#
#         # 设置 train 和 val
#         self.train = data
#         self.val = self.test
#
#         # 加载标签
#         self.test_labels = pd.read_csv(data_path + '/test_label.csv').values  # 标签只有一列
#
#         self.win_size_1 = win_size_1
#         self.count = count
#
#     def __len__(self):
#         if self.mode == "train":
#             return self.train.shape[0]
#         elif self.mode == 'val':
#             return self.val.shape[0]
#         elif self.mode == 'test':
#             return self.test.shape[0]
#         else:
#             return self.test.shape[0]
#
#     def __getitem__(self, index):
#         count = self.count
#
#         index = index * self.step
#
#         index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
#         index_1 = np.clip(index_1, 0, len(self.test) - 1)
#
#         data_block_1 = self.test[index_1]
#         index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
#                             index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
#         index_2 = np.clip(index_2, 0, len(self.test) - 1)
#         data_block_2 = self.test[index_2]
#         data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
#         return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class GHLSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # 加载训练数据
        data = np.load(data_path + "/GHL_train.npy")
        if data.ndim == 2:
            features_train = data[:, 1].reshape(-1, 1)  # 假设第二列是特征
        else:
            features_train = data.reshape(-1, 1)  # 如果只有一个特征列

        # 缩放特征
        self.scaler.fit(features_train)
        features_train = self.scaler.transform(features_train)

        # 加载测试数据
        test_data = np.load(data_path + "/GHL_test.npy")
        if test_data.ndim == 2:
            features_test = test_data[:, 1].reshape(-1, 1)  # 假设第二列是特征
        else:
            features_test = test_data.reshape(-1, 1)  # 如果只有一个特征列

        # 缩放测试特征
        self.test = self.scaler.transform(features_test)

        self.train = features_train
        self.val = self.test
        self.test_labels = np.load(data_path + "/GHL_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count

        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode == "val":
            return self.val.shape[0]
        elif self.mode == "test":
            return self.test.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        # 第一个数据块索引和提取
        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)
        data_block_1 = self.test[index_1]

        # 第二个数据块索引和提取
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, 1)  # 调整为适合单变量数据

        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class HAISegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count

        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class PUMPSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count

        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data

        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")[:, :]
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")[:, :]
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")[:]
        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class UCRSegLoader(object):
    def __init__(self, index, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.index = index
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/UCR_" + str(index) + "_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/UCR_" + str(index) + "_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/UCR_" + str(index) + "_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count
        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class DodgersSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step

        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/Dodgers" + "_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/Dodgers" + "_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/Dodgers" + "_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count
        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class UCRAUGSegLoader(object):
    def __init__(self, index, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.index = index
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/UCR_AUG_" + str(index) + "_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/UCR_AUG_" + str(index) + "_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/UCR_AUG_" + str(index) + "_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count
        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class NIPS_TS_WaterSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Water_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class NIPS_TS_SwanSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class NIPS_TS_CCardSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_CCard_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_CCard_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_CCard_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class WADISegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/WADI_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/WADI_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/WADI_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class SMD_OriSegLoader(object):
    def __init__(self, index, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step
        self.index = index
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_Ori_" + str(index) + "_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_Ori_" + str(index) + "_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMD_Ori_" + str(index) + "_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count
        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, win_size_1, count, step=1, flag="train"):
        self.mode = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        self.win_size_1 = win_size_1
        self.count = count
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


class YahooSegLoader(object):
    def __init__(self, data_path, win_size, win_size_1, count, step, mode="train"):
        self.mode = mode
        self.step = step

        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/Yahoo" + "_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/Yahoo" + "_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/Yahoo" + "_test_label.npy")
        self.win_size_1 = win_size_1
        self.count = count
        if self.mode == "val":
            print("train:", self.train.shape)
            print("test:", self.test.shape)

    def __len__(self):  ####更改
        if self.mode == "train":
            return (self.train.shape[0])
        elif (self.mode == 'val'):
            return (self.val.shape[0])
        elif (self.mode == 'test'):
            return (self.test.shape[0])
        else:
            return (self.test.shape[0])

    def __getitem__(self, index):
        count = self.count
        index = index * self.step

        index_1 = np.arange(index - self.win_size // 2, index + self.win_size // 2 + 1)
        index_1 = np.clip(index_1, 0, len(self.test) - 1)

        data_block_1 = self.test[index_1]
        index_2 = np.arange(index - self.win_size_1 // 2 - count // 2 * (self.win_size_1 + 1),
                            index + self.win_size_1 // 2 + 1 + count // 2 * (self.win_size_1 + 1))
        index_2 = np.clip(index_2, 0, len(self.test) - 1)
        data_block_2 = self.test[index_2]
        data_block_2 = data_block_2.reshape(-1, self.win_size_1 + 1, data_block_2.shape[1])
        return np.float32(data_block_1), np.float32(data_block_2), np.float32(self.test_labels[index])


def get_loader_segment(index, data_path, batch_size, win_size_1, count, win_size=100, step=1, mode='train',
                       dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'SWAT'):
        dataset = SWATSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'UCR'):
        dataset = UCRSegLoader(index, data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'UCR_AUG'):
        dataset = UCRAUGSegLoader(index, data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'NIPS_TS_Water'):
        dataset = NIPS_TS_WaterSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'NIPS_TS_Swan'):
        dataset = NIPS_TS_SwanSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'NIPS_TS_CCard'):
        dataset = NIPS_TS_CCardSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'SMD_Ori'):
        dataset = SMD_OriSegLoader(index, data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'HAI'):
        dataset = HAISegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'WADI'):
        dataset = WADISegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'SKAB'):
        dataset = SKABSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'Dodgers'):
        dataset = DodgersSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'GHL'):
        dataset = GHLSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'Genesis'):
        dataset = GenesisSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'PUMP'):
        dataset = PUMPSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    elif (dataset == 'Yahoo'):
        dataset = YahooSegLoader(data_path, win_size, win_size_1, count, 1, mode)
    
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8,
                             drop_last=True)


    return data_loader


def generate_time_slices(train, win_size):
    num_timestamps = train.shape[0]
    half_win_size = win_size // 2

    # 扩展 train 数组的两侧，以便处理边缘情况
    padded_train = np.pad(train, ((half_win_size, half_win_size), (0, 0)), mode='edge')

    # 构建索引数组，每个元素为中心时间戳的索引
    center_indices = np.arange(num_timestamps) + half_win_size

    # 构建时间片段索引数组
    slice_indices = np.arange(-half_win_size, half_win_size + 1)

    # 在索引数组上添加维度，以便进行广播
    center_indices = center_indices[:, np.newaxis]
    slice_indices = slice_indices[np.newaxis, :]

    # 计算时间片段的索引
    time_slice_indices = center_indices + slice_indices

    # 使用数组索引直接提取时间片段
    time_slices = padded_train[time_slice_indices]

    return time_slices
