
# Image-colorization using U-net and conditional GANs

The task of colourizing black and white photographs necessitates a lot of human input and hardcoding.
The goal of this project is to create an end-to-end deep learning pipeline that can automate the task of image colorization by taking a black and white image as input and producing a colourized image as output


## A Note Before Hand
I took this project this summer and it has been a great learning experience. It was such a fulfilling journey and I am really thankful for the guidance from the project mentors at VLG.

The Reader is free to try out the code I used in this project themselves by opening the google colab link provided below, train the models and look for the magic themselves.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
## A Preview

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Motivation

Colorization is the process of adding color information to monochrome photographs or videos. The colorization of grayscale images is an ill-posed problem, with multiple correct solutions. Deep learning algorithms that better understand image data like the colors that are generally observed for human faces should ideally perform better in this task.


### *IMAGE-COLORIZATION - BY Abhiyansh Raj** 
'Raja Harishchandra' was the first ever film in India, black and white  ofcourse. Indeed we have come a long way from those B & W days into SUPER-AMOLED crystal clear colourful times .Colours really bring out the emotion in a scene :)
Here's a project an attempt by me to make use of deep learning and see if I can successfully colorize a given black and white image to a colourdul one. 
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

Now, we will be making use of IMAGES from COCO dataset. This dataset has roughly 20000 images but for our purposes we will making use of roughly 10000.
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

## Creating the Dataset class
Here, we will resize the image and flip it horizontally(if it belongs to traing set, data augmentation technique) and convert the RGB image read into Lab colorspace where will separate the input(L) and output(Target - a and b).

-DATA AUGMENTATION : -
Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when training a machine learning model.

-RGB AND LAB :- 
RGB data represents colour in RGB colour space where there are 3 numbers for each pixel indicating how much Red, Green, and Blue the pixel is.
LAB on the other hand represents 3 channels - L channel , *a channel, *b channel where L represents lightness, and the other two encode how much green-red and yellow-blue each pixel is respectfully.

![download.jpeg](https://drive.google.com/uc?id=1tyBE2lUHJL2Z-W9U66FFEplBFjvykBMD)

*Credits for image - Graeme Cookson / Shutha.org*


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

let's see if this works:-

```python
train_dl = img_dataloaders(paths=train_paths, split='train')
val_dl = img_dataloaders(paths=val_paths, split='val')
```


## Acknowledgements

 - [LAB color space](http://shutha.org/node/851)
 - [GANs](https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09)
 - [CNN](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)
 - [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf)
 - [Conditional GANs](https://jonathan-hui.medium.com/gan-cgan-infogan-using-labels-to-improve-gan-8ba4de5f9c3d)
 - [Colorizing B&W images](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)

