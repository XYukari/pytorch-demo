import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize

from reader import gray_read, ycbcr_read


class Roadscene(Dataset):
    def __init__(self, ir_dir, vi_dir):
        self.ir_dir = ir_dir
        self.vi_dir = vi_dir
        self.transform = Resize((64, 64))
        self.ir_images = [f for f in os.listdir(self.ir_dir) if f.endswith('.jpg')]
        self.vi_images = [f for f in os.listdir(self.vi_dir) if f.endswith('.jpg')]

    def __len__(self):
        return min(len(self.ir_images), len(self.vi_images))

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Dataset index out of range")

        ir_image_path = os.path.join(self.ir_dir, self.ir_images[idx])
        vi_image_path = os.path.join(self.vi_dir, self.vi_images[idx])

        ir_image = gray_read(ir_image_path)
        vi_image_y, vi_image_cbcr = ycbcr_read(vi_image_path)

        ir_image = self.transform(ir_image)
        vi_image_y = self.transform(vi_image_y)
        vi_image_cbcr = self.transform(vi_image_cbcr)

        return ir_image, vi_image_y, vi_image_cbcr


def roadscene_loader(ir_dir, vi_dir, batch_size=4, num_workers=4):
    dataset = Roadscene(ir_dir, vi_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
