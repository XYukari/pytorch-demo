from pathlib import Path
from typing import Tuple

import cv2
import torch
from torch import Tensor
from kornia import image_to_tensor
from kornia.color import rgb_to_ycbcr, bgr_to_rgb


def gray_read(img_path: str | Path) -> Tensor:
    img_n = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_t = image_to_tensor(img_n).float() / 255
    return img_t


def ycbcr_read(img_path: str | Path) -> Tuple[Tensor, Tensor]:
    img_n = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_t = image_to_tensor(img_n).float() / 255
    img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
    y, cbcr = torch.split(img_t, [1, 2], dim=0)
    return y, cbcr
