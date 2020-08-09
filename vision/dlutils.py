import torch.nn as nn
import torchvision
import torch


def show_batch(batch_images: torch.Tensor):
    '''
    Function takes batch of images[BxCxHxW] and creates grid plot of it
    '''
    new_img = torchvision.utils.make_grid(
        batch_images).numpy().transpose(1, 2, 0)
    plt.imshow(new_img)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.(from fastai)"

    def __init__(self, sz: Optional[int] = None):
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)
