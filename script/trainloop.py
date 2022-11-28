import torch
# import tqdm as tqdm 
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score,auc,roc_auc_score

def train_loop(net, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to display losses and/or scores within the loop, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!

    Note: you don’t need to compute the score after each training iteration.
    If you do this, your training loop will be really slow!
    You should instead compute it every 50 or so iterations and aggregate ...
    """
    net.train()
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0
    p_labels = torch.tensor(()).to(device)
    y_labels = torch.tensor(()).to(device)
    
#     for i,data in enumerate(train_dataloader,0):
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for data in tepoch:
            optimizer.zero_grad()
            
            inputs, y_true = data['videos'].to(device), data['labels'].to(device)
            outputs = net(inputs)
            _,y_pred = torch.max(outputs,1)
            loss = criterion(outputs, y_true)
            loss.backward()
            optimizer.step()
        
            n = y_true.size(0)  #total number training example in batch
            LOSSES += loss.sum().data.cpu() * n        
            y_labels = torch.cat((y_labels, y_true), 0)
            p_labels = torch.cat((p_labels, y_pred), 0)
            tepoch.set_postfix(loss=loss.item())
            COUNTER += n
            # if COUNTER == 20:
                # break

        
        y_labels = y_labels.cpu()
        p_labels = p_labels.cpu()
        p_labels= p_labels.detach().numpy()
        y_labels = y_labels.detach().numpy()
    
        x = accuracy_score(y_labels, p_labels) #roc_auc_score#
        
    

        
    return x, LOSSES / float(COUNTER)

def valid_loop(net, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to display losses and/or scores within the loop, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!

    Note: you don’t need to compute the score after each training iteration.
    If you do this, your training loop will be really slow!
    You should instead compute it every 50 or so iterations and aggregate ...
    """


    net.eval()
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0
    p_labels = torch.tensor(()).to(device)
    y_labels = torch.tensor(()).to(device)
    with torch.no_grad():
        with tqdm(valid_dataloader, unit="batch") as tepoch:
#         for i,data in enumerate(valid_dataloader,0):
            for data in tepoch:
                inputs, y_true = data['videos'].to(device), data['labels'].to(device)
                outputs = net(inputs)
                _,y_pred = torch.max(outputs,1)
                loss = criterion(outputs, y_true)
#                 writer.add_scalar("Loss/validation", loss, epoch)
            
                n = y_true.size(0)  #total number training example in batch
                LOSSES += loss.sum().data.cpu() * n        
                y_labels = torch.cat((y_labels, y_true), 0)
                p_labels = torch.cat((p_labels, y_pred), 0)
                COUNTER += n
                
                # if COUNTER==20:
                    # break


    y_labels = y_labels.cpu()
    p_labels = p_labels.cpu()
    p_labels= p_labels.detach().numpy()
    y_labels = y_labels.detach().numpy()
    
    x = accuracy_score(y_labels, p_labels)
    return x, LOSSES / float(COUNTER)
