import torch, os, random
import torchvision.datasets
from torch.utils import data
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torchvision
from PIL import Image
import numpy as np
import torch.utils.data as data


class CKDataset(data.Dataset):
    '''
        used to train or test net
    Args:
        data_dir: the path of dataset
        mode: train\test
        fold: the k-fold cross validation
        transform: resize, totensor, flip
        ratio: the ratio of test and all data

        there are 135, 177, 75, 207, 84, 249, 54 images in data;
        we choose 123, 159, 66, 186, 75, 225, 48 images for train;
        we choose 12, 8, 9, 21, 9, 24, 6 images for test;
        the mode are in order according to the fold number.
    '''
    def __init__(self, data_dir=r"DataSet\CK+\emotion_cls", mode='train', ratio=0.1, fold=1, transform=None):
        self.transform = transform
        self.mode = mode   # train set or test set
        self.fold = fold   # the k-fold cross validation ---lwk,有待考虑
        self.ratio = ratio
        self.data_path = data_dir + "/train_test.txt"
        self.data_label_path = data_dir + "/labels.txt"
        self.data, self.label = self.load_data(self.data_path)
        self.labels = self.load_labels(self.data_label_path)
        image = []
        for i in range(len(self.data)):
            img = Image.open(os.path.join(data_dir, self.data[i]))
            pix_array = np.array(img)
            image.append(pix_array)
        image = np.array(image)

        # K-则交叉验证
        # number = len(self.data)  # 981
        # print(number)
        # sum_number = [0, 135, 312, 387, 594, 678, 927, 981]   # the sum of class number
        # test_number = [12, 18, 9, 21, 9, 24, 6]   # the number of each class
        # test_index = []
        # train_index = []
        # for j in range(len(test_number)):
        #     for k in range(test_number[j]):
        #         if self.fold != 10:   # the last fold start from the last element
        #             test_index.append(sum_number[j]+(self.fold-1)*test_number[j]+k)
        #         else:
        #             test_index.append(sum_number[j+1]-1-k)
        # for i in range(number):
        #     if i not in test_index:
        #         train_index.append(i)

        # now load the picked numpy arrays
        if self.mode == 'train':
            self.train_data = []
            self.train_labels = []
            for i in range(int(len(self.label)*(1-self.ratio))):
                self.train_data.append(image[i])
                self.train_labels.append(self.label[i])

        elif self.mode == 'test':
            self.test_data = []
            self.test_labels = []
            for i in range(int(len(self.label)*self.ratio)):
                self.test_data.append(image[len(self.label)-i-1])
                self.test_labels.append(self.label[len(self.label)-i-1])

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_labels[index]
        else:
            print("Error!!!")
        # H x W --> H x W x 3
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data), len(self.label)
        elif self.mode == 'test':
            return len(self.test_data), len(self.label)

    def load_data(self, input_data_path):
        with open(input_data_path, 'r') as f:
            data_file_list = f.read().splitlines()
            # Get full list of image and labels
            file_image = [s for s in data_file_list]
            file_label = [s.split('\\')[0] for s in data_file_list]
        return file_image, file_label

    def load_labels(self, input_data_path):
        with open(input_data_path, 'r') as f:
            data_list = f.read().splitlines()
            labels = [s.split('\t')[1] for s in data_list]
        return labels


class FERPlusDataset(data.Dataset):
    '''
        used to train or test net
    Args:
        mode: train or valid or test
        transform (callable, optional): A function/transforms that takes in an PIL image and returns a transformed version.
        there are 135, 177, 75, 207, 84, 249, 54 images in data;
        we choose 123, 159, 66, 186, 75, 225, 48 images for train;
        we choose 12, 8, 9, 21, 9, 24, 6 images for test;
        the mode are in order according to the fold number.
    '''
    def __init__(self, data_dir=r"./DataSet/FERPlus/data/emotion_cls", mode='train', ratio=0.1, fold=1, transform=None):
        self.transform = transform
        self.mode = mode   # train set or test set
        self.ratio = ratio
        self.data_test_path = data_dir + "/test.txt"
        self.data_train_path = data_dir + "/train.txt"
        self.data_valid_path = data_dir + "/valid.txt"
        self.data_label_path = data_dir + "/labels.txt"
        self.train_data, self.train_label = self.load_data(self.data_train_path)
        self.test_data, self.test_label = self.load_data(self.data_test_path)
        self.valid_data, self.valid_label = self.load_data(self.data_valid_path)
        self.labels = self.load_labels((self.data_label_path))

        if self.mode == 'train':
            self.image_train = []
            self.label_train = []
            for i in range(len(self.train_data)):
                img = Image.open(os.path.join(data_dir, self.train_data[i]))
                pix_array = np.array(img)
                self.image_train.append(pix_array)
                self.label_train.append(int(self.train_label[i]))
            image_train = np.array(self.image_train)
        elif self.mode == 'valid':
            self.image_valid = []
            self.label_valid = []
            for i in range(len(self.valid_data)):
                img = Image.open(os.path.join(data_dir, self.valid_data[i]))
                pix_array = np.array(img)
                self.image_valid.append(pix_array)
                self.label_valid.append(int(self.valid_label[i]))
            image_valid = np.array(self.image_valid)
        elif self.mode == 'test':
            self.image_test = []
            self.label_test = []
            for i in range(len(self.test_data)):
                img = Image.open(os.path.join(data_dir, self.test_data[i]))
                pix_array = np.array(img)
                self.image_test.append(pix_array)
                self.label_test.append(int(self.test_label[i]))
            image_test = np.array(self.image_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.image_train[index], self.label_train[index]
        elif self.mode == 'valid':
            img, target = self.image_valid[index], self.label_valid[index]
        elif self.mode == 'test':
            img, target = self.image_test[index], self.label_test[index]
        else:
            print("Error!!!")
        # H x W --> H x W x 3
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.image_train)
        elif self.mode == 'test':
            return len(self.image_test)
        elif self.mode == 'valid':
            return len(self.image_valid)

    def load_data(self, input_data_path):
        with open(input_data_path, 'r') as f:
            data_file_list = f.read().splitlines()
            # Get full list of image and labels
            file_image = [s.split('\t')[1] for s in data_file_list]
            file_label = [s.split('\t')[0] for s in data_file_list]
        return file_image, file_label

    def load_labels(self, input_data_path):
        with open(input_data_path, 'r') as f:
            data_list = f.read().splitlines()
            labels = [s.split('\t')[1] for s in data_list]
        return labels


class RAFDBDataset(data.Dataset):
    '''
        used to train or test net
    Args:
        mode: train or valid or test
        transform (callable, optional): A function/transforms that takes in an PIL image and returns a transformed version.
        there are 135, 177, 75, 207, 84, 249, 54 images in data;
        we choose 123, 159, 66, 186, 75, 225, 48 images for train;
        we choose 12, 8, 9, 21, 9, 24, 6 images for test;
        the mode are in order according to the fold number.
    '''
    def __init__(self, data_dir=r"DataSet\RAF-DB", mode='train', ratio=0.1, fold=1, transform=None):
        self.transform = transform
        self.mode = mode   # train set or test set
        self.ratio = ratio
        self.data_test_path = data_dir + "/train_list.txt"
        self.data_train_path = data_dir + "/train.txt"
        self.data_valid_path = data_dir + "/valid.txt"
        self.data_label_path = data_dir + "/labels.txt"
        self.train_data, self.train_label = self.load_data(self.data_train_path)
        self.test_data, self.test_label = self.load_data(self.data_test_path)
        self.valid_data, self.valid_label = self.load_data(self.data_valid_path)
        self.labels = self.load_labels((self.data_label_path))

        if self.mode == 'train':
            self.image_train = []
            self.label_train = []
            for i in range(len(self.train_data)):
                img = Image.open(os.path.join(data_dir, self.train_data[i]))
                pix_array = np.array(img)
                self.image_train.append(pix_array)
                self.label_train.append(int(self.train_label[i]))
            image_train = np.array(self.image_train)
        elif self.mode == 'valid':
            self.image_valid = []
            self.label_valid = []
            for i in range(len(self.valid_data)):
                img = Image.open(os.path.join(data_dir, self.valid_data[i]))
                pix_array = np.array(img)
                self.image_valid.append(pix_array)
                self.label_valid.append(int(self.valid_label[i]))
            image_valid = np.array(self.image_valid)
        elif self.mode == 'test':
            self.image_test = []
            self.label_test = []
            for i in range(len(self.test_data)):
                img = Image.open(os.path.join(data_dir, self.test_data[i]))
                pix_array = np.array(img)
                self.image_test.append(pix_array)
                self.label_test.append(int(self.test_label[i]))
            image_test = np.array(self.image_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.image_train[index], self.label_train[index]
        elif self.mode == 'valid':
            img, target = self.image_valid[index], self.label_valid[index]
        elif self.mode == 'test':
            img, target = self.image_test[index], self.label_test[index]
        else:
            print("Error!!!")
        # H x W --> H x W x 3
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.image_train)
        elif self.mode == 'test':
            return len(self.image_test)
        elif self.mode == 'valid':
            return len(self.image_valid)

    def load_data(self, input_data_path):
        with open(input_data_path, 'r') as f:
            data_file_list = f.read().splitlines()
            # Get full list of image and labels
            file_image = [s.split('\t')[1] for s in data_file_list]
            file_label = [s.split('\t')[0] for s in data_file_list]
        return file_image, file_label

    def load_labels(self, input_data_path):
        with open(input_data_path, 'r') as f:
            data_list = f.read().splitlines()
            labels = [s.split('\t')[1] for s in data_list]
        return labels


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((144, 144))
    ])
    # transform_train = None
    dataset = FERPlusDataset(mode='valid', transform=transform_train)
    for i in range(20):
        # print(dataset.__getitem__(i))
        dataset.__getitem__(i)
    print(dataset.__len__())
