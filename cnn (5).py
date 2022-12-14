import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
import time
import copy
import math


# Our model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 6)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Linear(64*10*10, 11)
        

    def forward(self, x, feature_conv=False):
        # 224x224
        x = self.pool(F.relu(self.conv1(x)))
        # 110x110
        x = self.pool(F.relu(self.conv2(x)))
        # 53x53
        x = self.pool(F.relu(self.conv3(x)))
        # 24x24
        x = self.pool(F.relu(self.conv4(x)))
        # 10x10
        if feature_conv:
            return x  
        x = x.view(-1, 64 * 10 * 10)
        x = self.fc(x)
        return x

    
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
    
# Class activation maps 
# batch-size has to be even for show_CAM
def returnCAM(feature_conv, weights_fc, class_idx):
    # generate the class activation maps upsample to 512x512
    # bs=batch size, nf=number of features, h= height, w=width
    transform = torchvision.transforms.Resize((512, 512))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs, nf, h, w = feature_conv.shape
    feature_conv = feature_conv.view(-1, nf * h * w)
    
    output_cam = []
    for i, idx in enumerate(class_idx):
        if not feature_conv.shape[1] == weights_fc.shape[1]:
            x = feature_conv[i].reshape((nf, h*w))
            x = torch.matmul(weights_fc[idx], x)
            x = x.reshape(1, h, w)
        else:
            x = torch.mul(feature_conv[i], weights_fc[idx])
            x = x.reshape(1, nf, h, w)
            x = torch.sum(x, 1)
            
        x = x - torch.min(x)
        x = x / torch.max(x)
        x = 255 * x
        x = transform(x)
        output_cam.append(x)
    return output_cam

def show_CAM(CAMs, orig_images, class_idx, classes, save_name):
    images = orig_images.cpu()
    bs, nf, h, w = images.shape
    images = np.transpose(images.numpy(), (0, 2, 3, 1))
    
    fig, axs = plt.subplots(2, int(math.ceil(bs/2)), figsize=(20,10))
    
    for i in range(bs):
        axs[int(i>=bs/2), int(i%(bs/2))].imshow(images[i])
        axs[int(i>=bs/2), int(i%(bs/2))].axis('off')
        
        if i < len(CAMs):
            heatmap = CAMs[i]
            heatmap = heatmap.cpu()
            heatmap = heatmap.detach().numpy()
            heatmap = np.squeeze(heatmap)
            axs[int(i>=bs/2), int(i%(bs/2))].imshow(heatmap, cmap='jet', alpha=0.5)
            axs[int(i>=bs/2), int(i%(bs/2))].set_title(classes[class_idx[i]])        
       
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    
# Dataset:
class CustomImageDataset:
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torchvision.io.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()