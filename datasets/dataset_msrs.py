import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import glob
import os
import cv2
from utils.common import RGB2YCbCr


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    filepath = glob.glob(os.path.join(data_dir, "*.bmp"))
    filepath.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    filepath.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    filepath.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    filepath.sort()
    filenames.sort()
    return filepath, filenames


class Generator(object):
    def __init__(self, output_size=None):  # 接收参数：output_size
        self.output_size = output_size

    def __call__(self, sample):
        image_vis, image_ir = sample['image_vis'], sample['image_ir']  # 从传入的字典 sample 中提取图像

        # 把image,label的shape调整到输入output_size
        if self.output_size:
            image_vis = cv2.resize(image_vis, (self.output_size[1], self.output_size[0]))
            image_ir = cv2.resize(image_ir, (self.output_size[1], self.output_size[0]))

        image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
        )  # 变为3*512*512并且归一化
        image_ir = np.asarray(Image.fromarray(image_ir), dtype=np.float32) / 255.0  # 512*512  归一化
        image_ir = np.expand_dims(image_ir, axis=0)  # 1*512*512
        image_vis = torch.from_numpy(image_vis)
        image_ir = torch.from_numpy(image_ir)
        image_vis_y, image_vis_cb, image_vis_cr = RGB2YCbCr(image_vis)
        sample = {'image_vis_y': image_vis_y, 'image_ir': image_ir, 'image_vis_cb': image_vis_cb,
                  'image_vis_cr': image_vis_cr}
        return sample


class MSRS_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):  # base_dir=../MSRS
        self.data_dir = base_dir
        assert split in ['train', 'test'], 'split must be "train"|"test"'
        self.transform = transform

        if split == 'train':
            self.data_dir_vis = os.path.join(self.data_dir, 'train', 'vi')
            self.data_dir_ir = os.path.join(self.data_dir, 'train', 'ir')

        elif split == 'test':
            self.data_dir_vis = os.path.join(self.data_dir, 'test', 'vi')
            self.data_dir_ir = os.path.join(self.data_dir, 'test', 'ir')

        self.filepath_vis, self.filenames_vis = prepare_data_path(self.data_dir_vis)
        self.filepath_ir, self.filenames_ir = prepare_data_path(self.data_dir_ir)

        self.split = split
        self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        image_vis = np.array(Image.open(vis_path))  # 480*640*3
        image_ir = Image.open(ir_path).convert('L')
        image_ir = np.array(image_ir)  # 480*640

        sample = {'image_vis': image_vis, 'image_ir': image_ir}
        if self.transform:  # 就是在这调用上面的Generator
            sample = self.transform(sample)
        sample['name'] = self.filenames_vis[index]
        return sample

    def __len__(self):
        return self.length
