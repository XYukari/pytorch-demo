import os

import torch
from PIL import Image
from torch.utils.data import DataLoader

from loader.roadscene import Roadscene
from model import FusionNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ir_dir = 'data/roadscene/cropinfrared'
vi_dir = 'data/roadscene/crop_HR_visible'
dataset = Roadscene(ir_dir, vi_dir)
dataloader = DataLoader(dataset, batch_size=4)

model = FusionNet().to(device)
model.eval()

save_dir = './save_dir'
id = 0

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, (ir_image, vi_image_y, vi_image_cbcr) in enumerate(dataloader):
    ir_image = ir_image.to(device)
    vi_image_y = vi_image_y.to(device)
    vi_image_cbcr = vi_image_cbcr.to(device)

    with torch.no_grad():
        output = model(ir_image, vi_image_y)

    fusion = torch.cat((output, vi_image_cbcr), dim=1)
    # fusion = output

    fusion = fusion.clamp(0, 1)

    fusion_image = fusion.permute(0, 2, 3, 1)
    fusion_image = fusion_image.numpy()

    images = []
    for i in range(fusion_image.shape[0]):
        img = Image.fromarray(fusion_image[i], 'YCbCr')
        images.append(img)

    for i, img in enumerate(images):
        filename = f'{id}.jpg'
        id += 1
        full_path = os.path.join(save_dir, filename)
        img.save(full_path)
        print(f'Saved {filename} to {full_path}')
