
# Image-colorization using U-net and conditional GANs

The task of colourizing black and white photographs necessitates a lot of human input and hardcoding.
The goal of this project is to create an end-to-end deep learning pipeline that can automate the task of image colorization by taking a black and white image as input and producing a colourized image as output


## A Note Before Hand
I took this project this summer and it has been a great learning experience. It was such a fulfilling journey and I am really thankful for the guidance from the project mentors at VLG.

The Reader is free to try out the code I used in this project themselves by opening the google colab link provided below, train the models and look for the magic themselves.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b_1Llx12pKGMaUXPDz90Ar8CXlNUfUUh?authuser=1#scrollTo=DV9iVxPpIgDI)

I have provided fundamental code snippets in this README, for complete code please refer to the colab link.
## A Preview

![App Screenshot](https://i.imgur.com/7sJGzK4.png)


## Motivation

Colorization is the process of adding color information to monochrome photographs or videos. The colorization of grayscale images is an ill-posed problem, with multiple correct solutions. Deep learning algorithms that better understand image data like the colors that are generally observed for human faces should ideally perform better in this task.


### *IMAGE-COLORIZATION - BY Abhiyansh Raj*
'Raja Harishchandra' was the first ever film in India, black and white  ofcourse. Indeed we have come a long way from those B & W days into SUPER-AMOLED crystal clear colourful times .Colours really bring out the emotion in a scene :)
Here's a project an attempt by me to make use of deep learning and see if I can successfully colorize a given black and white image to a colourful one. 
Hope you like my work, I'll be adding my thoughts and lessons in between the code wherever I feel there's a point worth noting.

Below are the libraries and frameworks we'll be requiring, the code is mainly PyTorch but we will be requiring fastai at a few places to help us build things without complications.

```python
import os
import glob
import numpy as np
import time
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb # we will convert the rgb image into an LAB format 
from tqdm.notebook import tqdm 

import torch
from torch import nn, optim
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_colab = None
```

```python
!pip install fastai==2.4
```

### OUR DATASET :

Now, we will be making use of IMAGES from COCO(Common Object in Context) dataset. This dataset has roughly 20000 images but for our purposes we will making use of roughly 10000.
In which, 8000 will be used for training and the rest 2000 as our validation set.

```python
from fastai.data.external import untar_data, URLs
data_path = untar_data(URLs.COCO_SAMPLE)
data_path = str(data_path) + "/train_sample"
use_colab = True
```
```python
if use_colab == True:
    path = data_path
else:
    path = "Your path to the dataset"# if locally
    
paths = glob.glob(path + "/*.jpg") # IMAGE FILES
np.random.seed(123)#to generate same random state numbers

paths_subset = np.random.choice(paths, 10_000, replace=False) # 10000 images out of 20000 randomly
rand_idxs = np.random.permutation(10_000)

train_idxs = rand_idxs[:8000] # 8000 - training set
val_idxs = rand_idxs[8000:] # 2000 - validation set

train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]

print(len(train_paths), len(val_paths))
```
Now, let's take a look at a few images from the set as examples

```python
_, axes = plt.subplots(5, 5, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")
```

![](https://i.imgur.com/yn74xKE.png)

## Creating the Dataset class
Here, we will resize the image and flip it horizontally(if it belongs to traing set, data augmentation technique) and convert the RGB image read into Lab colorspace where will separate the input(L) and output(Target - a and b).

-DATA AUGMENTATION : -
Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when training a machine learning model.

![](https://i.imgur.com/fQXOOn8.jpg)

-RGB AND LAB :- 
RGB data represents colour in RGB colour space where there are 3 numbers for each pixel indicating how much Red, Green, and Blue the pixel is.
LAB on the other hand represents 3 channels - L channel , *a channel, *b channel where L represents lightness, and the other two encode how much green-red and yellow-blue each pixel is respectfully.

![download.jpeg](https://drive.google.com/uc?id=1tyBE2lUHJL2Z-W9U66FFEplBFjvykBMD)

*Credits for image - Graeme Cookson / Shutha.org*

When utilising L*a*b, we can provide the model with the L channel (the grayscale image) and ask it to forecast the other two channels (*a, *b). After it makes its prediction, we concatenate all the channels to produce the coloured image. However, if you use RGB, you must first convert your image to grayscale, then feed the grayscale image to the model and hope it predicts three numbers for you. This is a much trickier and unstable task because there are far more combinations of three numbers than there are of two numbers, making the task much more difficult and unstable.


```python
#MAKING THE DATACLASS

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), #Data augmentation technique
            ])
        elif split == 'val': # if the image belongs to validation set we do not flip it
            self.transforms = transforms.Resize((256, 256),  Image.BICUBIC)
        
        self.split = split
        self.size = 256
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to LAB colour space as told above
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # scaling the values between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)

#creating the dataloader

def img_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):
    dataset = ColorizationDataset(**kwargs) # autocall to the dataset class
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
```

### LOSSES IN OUR model
Here x is the grayscale image, z is the generator's input noise, and y as the desired 2-channel output (it can also represent the 2 colour channels of a real image). Additionally, D is the discriminator and G is the generating model.
The noise is introduced in the form of *dropout layers*.
![](https://i.imgur.com/ffB7vgl.jpg)

Mean Absolute Error
![](https://i.imgur.com/5AmApyf.jpg)

Combining the adversial loss and mae.
![](https://i.imgur.com/Etlfmp0.jpg)

# Making the Generator
Here we make use of U-Net architecture in the generator of our GAN.

![](https://i.imgur.com/JsOwaI2.png)
Working : - The U-Net adds down-sampling and up-sampling modules to the left and right of that middle module (respectively) at every iteration until it reaches the input module and output module.

```python
class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, inplace = True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)
```

## Making the **Discriminator**
Now, below is the code for our discriminator. Here, we are stacking blocks of Conv-BatchNorm-LeackyRelu to make a decision about the reality or fakeness of our input image.

-We will not apply normalization to first and last blocks and the last block will have no activation function too.

### **Patch Discriminator**
In a normal discriminator, the model only gives out 1 number which represents what the model thinks of the input as in the whole image whether it is real or fake. On the other hand, our patch disciminator works on every patch like 50 by 50 pixels of the input image and for each patch it decides whether it is fake or not.

```python
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)] # the 'if' statement ensures not using stride of 2 for last block
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # No normalization or activation for last layer
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
```




## Results and Visualization
Now,let us get a sense of our model functioning till this stage.
After about 40 -50 epochs, we obtain one of the results as follows.

![](https://i.imgur.com/5loW9wh.png)

Though there is some promise, but the results are far from perfect. Notice, how the model fills areas with just gray or brown when its unsure what to predict.

## THROUGH GRAPHS


![](https://i.imgur.com/K8Zo2DR.png)

The convergence of false and actual images toward one another, which is visible, suggests that the generator is becoming more adept at creating real images.

![](https://i.imgur.com/993uNr6.png)

If trained on more epochs, the generator loss will further decrease as the number of epochs increases.

![](https://i.imgur.com/z2IFNiQ.png)

*Generator v/s discriminator loss.*

Indeed, we have come a long way and the results we have obtained our good(after about 50 epochs) but as we can see , the model still does not recognize the colour of some objects. Let us now apply a different approach.

# NEW MODEL
Here, first we are going to pre train the generator in a superwised manner.
This will be done in 2 stages: Stage 1: - The backbone of the generator (the down sampling path) is a pretrained model for classification (on ImageNet) 2- The whole generator will be pretrained on the task of colorization with L1 loss.

We will be using a pretrained ResNet18 as the backbone of our U-Net and further we will train that U-Net on our training set with L1 Loss. Last but not least we will have the combined adversial and L1 loss as done earlier.

## New Generator
We will use fastai (as mentioned earlier) library's Dynamic U-Net module to avoid complicated stuff.
```python
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
```
```python
def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)  # Ignoring the last two layers (GlobalAveragePooling and a Linear layer)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G
```
### Pre-Training the generator
```python
def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")

net_G = build_res_unet(n_input=1, n_output=2, size=256)
opt = optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()        
pretrain_generator(net_G, train_dl, opt, criterion, 20)
torch.save(net_G.state_dict(), "res18-unet.pt") #saving pretrained model
# we will train this for about 20 epochs
```
# OUR FINAL MODEL
Now,let us put our new changes into effect.

```python
def train_model(model, train_dl, epochs):
    valid_data = next(iter(val_dl))        # Visualizing output on valid_Data

    for e in range(epochs):
  
        loss_meter_dict = create_loss_meters() 
                                         
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0))        # Updating the loss_meter_dict
        
        model_save_name = f"imgcl_model_final_{e+1}.pt"
        path = f"/content/gdrive/My Drive/Image-colouriziation/{model_save_name}"      # Saving the model after every epoch
        torch.save(model.state_dict(), path)


        print(f"\nEpoch {e+1}/{epochs}")
        log_results(loss_meter_dict) # Function to print out the losses
        visualize(model, valid_data, e) # Function displaying the model's outputs
 ```
        
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
model = MainModel(net_G=net_G)
train_model(model, train_dl, 20)
```

# **RESULTS, COMPARISIONS AND FINDINGS**
So, afters hours of traing and tweaking here we are with the final results.
Notice, the improvement we have over our previous results. The colours are more realistic and to the point. Not only the second strategy improved our results but also took less time and epochs to train.

![](https://i.imgur.com/2syAp6Y.png)

another set

![](https://i.imgur.com/L6Fnd8e.png)


## The case of DROPOUT
IN the begininning, in our generator architecture, the cause of the noise was the dropout layers but in the U-Net we built with the help of fastai had no dropout layers !

What does this imply ? 
The conditional GAN is still able to function without the dropout layers however the outputs will be more deterministic because of the lack of that noise but  the grayscale image has still crucial information sufficient enough for the generator to produce convincing outputs. 
But,in reality the adversarial training does give us an edge and is more fruiful.


# CONCLUSIONS
We have successfully achieved our task of colourizing black and white images. The results clearly show a level of improvement. The final prediction is very close to the actual image, thus pretraining our generator was really helpful.

## AN END Note

With this we have achieved our objective. I would again like to state that this project was a wonderful experience and I am really thankful to VLG for this opportunity. Hoping to collaborate more in the future !


## Contributions

Contributions are always welcome!



## Acknowledgements

 - [LAB color space](http://shutha.org/node/851)
 - [GANs](https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09)
 - [CNN](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)
 - [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf)
 - [Conditional GANs](https://jonathan-hui.medium.com/gan-cgan-infogan-using-labels-to-improve-gan-8ba4de5f9c3d)
 - [Colorizing B&W images](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)

