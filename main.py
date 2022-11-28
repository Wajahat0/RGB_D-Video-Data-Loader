
import os
import argparse
import torchvision
from tqdm import tqdm
# import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
from sklearn.metrics import accuracy_score,auc,roc_auc_score
from script.video_data_loader import VideoDataLoader
import pytorchvideo.models.resnet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from script.get_model import get_model
from script.trainloop import valid_loop, train_loop


parser = argparse.ArgumentParser(description='Model_training.')
parser.add_argument("-r_dir", "--root_dir", help="Enter the path to root dir.", type = str, default=r"C:\Users\wajah\Desktop\Training")
# parser.add_argument("-d_dir", "--data_dir", help="Enter the path to data dir data\train.", default=r"RGB_D\RGB")
parser.add_argument("-batch_size", "--batch_size", help="Enter the batch_size.",type=int, default=10)
parser.add_argument("-fps", "--frame_rate", help="Enter the desired frame rate.",type=int, default=15)
parser.add_argument("-clip_d", "--clip_duration", help="Enter the desired frame rate.",type=int, default=30)
parser.add_argument("-num_classes", "--num_classes", help="Enter Number of classes.",type=int, default=60)
parser.add_argument("-model", "--model", help="Select from the list ['slow_r50' ,'c2d_r50' ,'r3d_18','slowfast_r50','r2plus1D_18']",type = str, default='r3d_18')
parser.add_argument("-weights", "--pre_trained", help="Should model use pretrained weights are not True/False.",type=bool, default=True)
parser.add_argument("-m", "--mode", help="Data_type the microscope: RGB, Depth,RGB_D.",type = str, default="RGB")
parser.add_argument("-lr", "--learning_rate", help="Learning rate screach  0.01, 0.001 transfer Learning",type=float, default=0.001)
parser.add_argument("-ep", "--epochs", help="Learning rate screach  0.01, 0.001 transfer Learning", type=int, default=2)

args = parser.parse_args()
# """
#         python --batch_size 2   
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#             python main.py --root_dir C:\Users\wajah\Desktop\Training --batch_size 2 --mode RGB_D
        
#     """


batch_size = args.batch_size
num_frames = args.clip_duration
num_epochs = args.epochs
frame_rate = args.frame_rate
num_classes = args.num_classes
pre_trained= args.model
weight =args.pre_trained
lr= args.learning_rate
mode = args.mode
path_to_folder= args.root_dir
# data_folder = args.data_dir

model_path  = os.path.join(args.root_dir,'model',  (args.model + '_'+str(args.clip_duration) + str(args.frame_rate)+'.pth'))
result_path = os.path.join(args.root_dir,'result',  (args.model +'_' +str(args.clip_duration) + str(args.frame_rate)+'.npy'))


print(model_path)
print(result_path) 

train_dataset = VideoDataLoader(root_dir=path_to_folder,csv_dir='data_files/RGB_D_train_1.csv', 
                                transform=True,frame_rate=frame_rate, image_size=112, num_frames = num_frames,
                                mode=mode)

test_dataset = VideoDataLoader(root_dir=path_to_folder,csv_dir='data_files/RGB_D_test_1.csv',
                               transform=True,frame_rate=frame_rate, image_size=112, num_frames = num_frames,
                               mode=mode)


print(len(train_dataset))


train = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=0,pin_memory=True)
test = DataLoader(test_dataset, batch_size=batch_size,shuffle=True, num_workers=0,pin_memory=True)
for x in train:
    print(x["videos"].shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net= get_model(pretrained_model=pre_trained,num_classes=num_classes, weight=weight,mode=args.mode)
net.to(device)

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)


v_loss= np.array(10.0)

train_losses, valid_losses = [], []
train_accuracy, valid_accuracy = [], []
for epoch in range(num_epochs):
    tqdm.write(f"====== Epoch {epoch} ======>")
    
    train_acc,train_loss = train_loop(net, train, device, optimizer, criterion)
    valid_acc,valid_loss = valid_loop(net, test, device, optimizer, criterion)
    train_losses.append(train_loss)
    train_accuracy.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accuracy.append(valid_acc)
    if v_loss>valid_loss.cpu().numpy():
        v_loss=valid_loss.cpu().numpy()
        torch.save(net,model_path)
        print('validation loss improved and new model is saved')
    
    
    print('validation_loss',valid_loss)
    curr_lr = optimizer.param_groups[0]['lr']
    
    print(f'Epoch {epoch}\ \
        Training Loss: {train_loss}\ \
        Training Accuracy: {train_acc}\ \
        Validation Loss:{valid_loss}\ \
        Validation Loss:{valid_acc}\ \
        LR:{curr_lr}')

    scheduler.step()

print(f"===== Best validation loss: {min(valid_losses):.3f} =====>")
print(f"===== Best validation Accuracy: {max(valid_accuracy):.3f} =====>") 

loss= []
np.save(result_path,np.array(loss.append([train_losses, train_accuracy,valid_losses, valid_accuracy])))
print('training Done')