import torch
import numpy as np
from numpy import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class ToTensor(object):

    def __init__(self, num_frames,frame_rate):
        assert isinstance(num_frames, (int, tuple))
        assert isinstance(frame_rate, (int, tuple))
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.step = int(30/self.frame_rate)

    def __call__(self, sample):


        image, landmarks = sample['videos'], sample['labels']
        t_frames= image.shape[1] #total number of frames
        hello = self.step*self.num_frames
        while image.shape[1]<=hello:
            image = np.concatenate((image,image[:,-1:,:,:]),axis=1)
        t_frames= image.shape[1] 

        try: 
            start = random.randint(0, t_frames-self.step*self.num_frames)
        except:
                start = 0        
        frames = list(range(start,start+self.num_frames*self.step,self.step))
        return {'videos': torch.from_numpy(image[:,frames,:,:]),
                'labels': landmarks}


class CenterCrop_resize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, output_size))
        # assert isinstance(frame_rate, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):


        video, landmarks = sample['videos'], sample['labels']

        frames = torch.FloatTensor(video.shape[0], video.shape[1], self.output_size, self.output_size)

        h,w  = video[:,0,:,:].shape[-2:]
        if h>w:
            crop_size=w
        else:
            crop_size=h

        for i in range(video.shape[1]):
            image = video[:,i,:,:]    
            image = T.CenterCrop(size=crop_size)(image)
            image = T.Resize(size=self.output_size)(image)
            frames[:,i,:,:] =image
            
        return {'videos': frames,
                'labels': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        video, landmarks = sample['videos'], sample['labels']
        frames = torch.FloatTensor(video.shape[0], video.shape[1], self.output_size,self.output_size)
        h, w = video[:,0,:,:].shape[-2:]
        new_h = self.output_size

        new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        for i in range(video.shape[1]):
            
            image = video[:,i,:,:] 
            image = image[:,top: top + new_h,left: left + new_w]
            frames[:,i,:,:] =image

        return {'videos': frames, 'labels': landmarks}

class RandomRotate(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, range):
        assert isinstance(range, (int, tuple))
        self.range = range

    def __call__(self, sample):
        
        video, landmarks = sample['videos'], sample['labels']

        if random.random() > 0.5:
            frames = torch.FloatTensor(video.shape[0], video.shape[1], video.shape[2],video.shape[3])
            angle = random.randint(-self.range, self.range)
            for i in range(video.shape[1]): 
                frames[:,i,:,:] =TF.rotate(video[:,i,:,:], angle)

            return {'videos': frames, 'labels': landmarks}
        else:
            return {'videos': video, 'labels': landmarks}

class RandomHorizontalFlip(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, range):
        assert isinstance(range, (float, tuple))
        self.range = range

    def __call__(self, sample):
        video, landmarks = sample['videos'], sample['labels']
        if random.random() > self.range:
            frames = torch.FloatTensor(video.shape[0], video.shape[1], video.shape[2],video.shape[3])
            for i in range(video.shape[1]):  
                frames[:,i,:,:] =T.RandomHorizontalFlip(p=1)(video[:,i,:,:])

            return {'videos': frames, 'labels': landmarks}
        else:
            return {'videos': video, 'labels': landmarks}

class RandomVerticalFlip(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, range):
        assert isinstance(range, (float, tuple))
        self.range = range

    def __call__(self, sample):
        video, landmarks = sample['videos'], sample['labels']

        if random.random() > self.range:
            frames = torch.FloatTensor(video.shape[0], video.shape[1], video.shape[2],video.shape[3])
            for i in range(video.shape[1]):  
                frames[:,i,:,:] = T.RandomVerticalFlip(p=1)(video[:,i,:,:])

            return {'videos': frames, 'labels': landmarks}
        else:
            return {'videos': video, 'labels': landmarks}














