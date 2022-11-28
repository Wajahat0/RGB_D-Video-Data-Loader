import torch
import cv2
import numpy as np
def read_videos(path_2_file):

    cap = cv2.VideoCapture(path_2_file)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = torch.FloatTensor(3, num_frames, height, width)
    ret=True
    frame_num=0
    while ret:
        ret, img = cap.read()
        if ret==False:
            cap.release()
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            frames[:, frame_num, :, :] = img.float()
            frame_num += 1
    frames /= 255
    return frames.numpy()