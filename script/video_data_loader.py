from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from numpy import random
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms, utils
from script.read_videos import read_videos
from script.transformation import ToTensor, CenterCrop_resize, RandomCrop, RandomRotate,RandomHorizontalFlip,RandomVerticalFlip

import warnings
warnings.filterwarnings("ignore")


class VideoDataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir,csv_dir, transform=None, frame_rate=30, image_size=112, num_frames = 15,mode='RGB'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.C_data_dir= 'RGB_D/RGB'
        self.D_data_dir= 'RGB_D/Depth'
        # self.csv_dir = r'data_files\RGB_D_train_1.csv'
        self.rgb_root_dir = os.path.join(root_dir,self.C_data_dir)
        self.depth_root_dir = os.path.join(root_dir,self.D_data_dir)
        print(self.rgb_root_dir)
        print(self.depth_root_dir)
        self.landmarks_frame = pd.read_csv(os.path.join(root_dir,csv_dir))
        self.transform = transform
        self.num_frames = num_frames
        self.image_size = image_size
        self.channels = 3 
        self.frame_rate = frame_rate
        self.transforms_norm = transform
        self.transform=transform
        self.mode=mode


    def __len__(self):
        return len(self.landmarks_frame)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        img_name = os.path.join(self.rgb_root_dir,
                                self.landmarks_frame.iloc[idx, 1])
        depth = os.path.join(self.depth_root_dir,self.landmarks_frame.iloc[idx, 1])
        # frames=read_videos(img_name)
        if self.mode =='Depth':
            # frames=read_videos(img_name)
            xz= np.load(depth[:-8]+'.npz')
            frames =xz['a']
        elif self.mode =='RGB_D':
            frames=read_videos(img_name)
            # print(depth[:-8]+'.npz')
            xz= np.load(depth[:-8]+'.npz')
            frames = np.concatenate((frames,xz['a']))#,frames[0:1,:,:,:]))
        else:
            frames=read_videos(img_name)
            # print('helo')
        # frames = np.array(self.load_video_frame(img_name)).reshape(self.channels, self.num_frames, self.image_size, self.image_size)
        # frames = frames.reshape(self.channels, self.num_frames, self.image_size, self.image_size)
        sample = {'videos': frames, 'labels': np.array(self.landmarks_frame.iloc[idx, 2])}
        sample=self.transforms_norm(sample)
        return sample

