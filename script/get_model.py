import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, r2plus1d_18, R3D_18_Weights
from torch.fx import symbolic_trace

import warnings
warnings.filterwarnings("ignore")
def get_model(pretrained_model='slow_r50',num_classes=50, weight=True, mode='RGB'):
    if pretrained_model == 'c2d_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'c2d_r50', pretrained=weight)
        # change the classification layer from 400 to 50
        model.blocks[6].proj= nn.Linear(2048, num_classes)

        if mode == 'RGB_D':
            model.blocks[0].conv = nn.Conv3d(4, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False) 
        elif mode=='Depth':
            model.blocks[0].conv = nn.Conv3d(1, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        else:
            pass

    elif pretrained_model == 'slow_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=weight)
        # change the classification layer from 400 to 50
        model.blocks[5].proj= nn.Linear(2048, num_classes)
        
        if mode == 'RGB_D':
            model.blocks[0].conv = nn.Conv3d(4, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False) 
        elif mode=='Depth':
            model.blocks[0].conv = nn.Conv3d(1, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        else:
            pass

    elif pretrained_model == 'slowfast_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=weight)
        # change the classification layer from 400 to 50
        model.blocks[6].proj= nn.Linear(2048, num_classes)
        
        if mode == 'RGB_D':
            model.blocks[0].multipathway_blocks[0].conv = nn.Conv3d(4, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
            model.blocks[0].multipathway_blocks[1].conv = nn.Conv3d(4, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False) 
        elif mode=='Depth':
            model.blocks[0].multipathway_blocks[0].conv = nn.Conv3d(1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
            model.blocks[0].multipathway_blocks[1].conv = nn.Conv3d(1, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        else:
            pass

    elif pretrained_model == 'r3d_18':
        # weights = R3D_18_Weights.DEFAULT
        model = r3d_18(pretrained=True)
        # change the output layer from 400 to 50
        model.fc= nn.Linear(512, num_classes) #nn.Identity()
        if mode == 'RGB_D':
                model.stem[0] = nn.Conv3d(4, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False) 
        elif mode=='Depth':
                model.stem[0] = nn.Conv3d(1, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        else:
            pass

    elif pretrained_model == 'r2plus1d_18':
        model = r2plus1d_18(pretrained=weight)
        model.fc= nn.Linear(512, num_classes)
        if mode == 'RGB_D':
            model.stem[0] = nn.Conv3d(4, 45, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False) 
        elif mode=='Depth':
            model.stem[0] = nn.Conv3d(1, 45, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        else:
            pass
    else: 
        print('worng input model select only from the list "[slow_r50 ,c2d_r50 ,r3d_18,slowfast_r50,r2plus1D_18]"')
    return model

