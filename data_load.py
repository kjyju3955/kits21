import os
import numpy as np
from glob import glob
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class Datasets(torch.utils.data.Dataset):
    def __init__(self, dir_path, start, end, transform=None, mode="AND"):
        self.transform = transform
        self.mode = mode
        self.dir_path = dir_path
        self.start = start
        self.end = end

        self.input_dic = []
        self.label_dic = []
        self.make_dict()

    def select_mode(self):
        path_dict = {
            'AND': glob(os.path.join(self.dir_path, 'aggregated_AND_seg.nii.gz')),
            'OR': glob(os.path.join(self.dir_path, 'aggregated_OR_seg.nii.gz')),
            'MAJ': glob(os.path.join(self.dir_path, 'aggregated_MAJ_seg.nii.gz'))
        }

        return path_dict[self.mode]

    def one_hot(self, label, one_hot_arr, labels=np.array([1, 2, 3, 4])):
        for i in range(0, 4):
            seg_ = label == labels[i]
            one_hot_arr[i, :, :] = seg_[0:label.shape[0], 0:label.shape[1]]

        return one_hot_arr

    def data_plot(self, input_data, label_data):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(input_data[input_data.shape[0] // 2], cmap="Greys")
        ax1.set_title('Image')
        ax2.imshow(label_data[label_data.shape[0] // 2])
        ax2.set_title('Mask')
        plt.show()

        # plt.hist(input_data.ravel(), bins=100, density=True, color='b', alpha=1)
        # plt.show()

    def normalize_img(self, img):
        img = img.astype("float32")
        min_perc, max_perc = np.percentile(img, 5), np.percentile(img, 95)
        img_valid = img[(img > min_perc) & (img < max_perc)]
        mean, std = img_valid.mean(), img_valid.std()
        img = (img - mean) / std
        return img

    def get_data(self, num):
        input_path = glob(os.path.join(self.dir_path, 'imaging.nii.gz'))
        label_path = self.select_mode()

        input_data = nib.load(input_path[num]).get_fdata()  # numpy 형식으로 data get
        label_data = nib.load(label_path[num]).get_fdata()

        for i in input_data:
            i = self.normalize_img(i)  # clipping norm
            i = i[np.newaxis, :, :]  # 3차원으로 변경
            i = torch.from_numpy(i)
            self.input_dic.append(i)


        for i in label_data:
            one_hot_arr = np.zeros((len(np.array([0, 1, 2, 3])), 512, 512))
            #i = self.one_hot(i, one_hot_arr)  # 4가지로 one-hot encoding + 3차원으로 변환
            #i = torch.from_numpy(i)
            self.label_dic.append(self.one_hot(i, one_hot_arr))
        # print(self.label_dic[0].shape)

        self.data_plot(input_data, label_data)  # data plot

    def make_dict(self):
        for i in range(self.start, self.end):
            self.get_data(i)

    def __len__(self):
        return len(self.label_dic)

    def __getitem__(self, index):
        data = {'input': torch.tensor(np.array(self.input_dic[index])),
                'label': torch.tensor(self.label_dic[index])}

        return data


if __name__ == "__main__":
    dir_path = './kits21/data/case_00*'
    test = Datasets(dir_path, start=0, end=1)
    test.make_dict()
'''

if __name__ == "__main__":
    # image_path = './kits21/data/case_00*'

    # dataset = Datasets(image_path)

    print(max(np.unique(image.get_fdata())))
    print(min(np.unique(image.get_fdata())))

    print(np.unique(label.get_fdata()))'''

'''test_image = nib.load(input_path).get_fdata()
test_mask = nib.load(label_path).get_fdata()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(test_image[test_image.shape[0] // 2])
ax1.set_title('Image')
ax2.imshow(test_mask[test_image.shape[0] // 2])
ax2.set_title('Mask')

plt.show()'''
