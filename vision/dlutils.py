
import mimetypes
import os
import torch.nn as nn
import torchvision
import torch
import cv2
from pathlib import Path
import pandas as pd

all = ["show_batch", "AdaptiveConcatPool2d", "get_files"]


# Cell
def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

# Cell


def get_files(path, extensions=None, recurse=True, folders=[], followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders = folders
    extensions = extensions
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        # returns (dirpath, dirnames, filenames)
        for i, (p, d, f) in enumerate(os.walk(path, followlinks=followlinks)):
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) != 0 and i == 0 and '.' not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return list(res)


def show_batch(batch_images: torch.Tensor):
    '''
    Function takes batch of images[BxCxHxW] and creates grid plot of it
    '''
    new_img = torchvision.utils.make_grid(
        batch_images).numpy().transpose(1, 2, 0)
    plt.imshow(new_img)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.(from fastai)"

    def __init__(self, sz: int = None):
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


# class ImageDataset(Dataset):
#     """__init__ and __len__ functions are the same as in TorchvisionDataset"""

#     def __init__(self, file_paths: Path, df: pd.DataFrame, transform=None):
#         '''
#         file_path = Path('./data)
#         df= data_frame

#         '''
#         self.image_paths = file_paths
#         self.df = df
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):

#         file_path = self.file_path / self.df.iloc[idx]['File']
#         label = self.df.iloc[idx]['Label']

#         # Read an image with OpenCV
#         image = cv2.imread(file_path)

#         # By default OpenCV uses BGR color space for color images,
#         # so we need to convert the image to RGB color space.
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented['image']

#         return image, label
