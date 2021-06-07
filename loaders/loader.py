import math as m

import numpy as np
import torch
from torchvision.transforms import ToTensor
from skimage import color, morphology, filters
from PIL import Image

from utils.runner_utils import normalize, denormalize


class ImageHolder:
    def __init__(self,
                 img_path,
                 resampler,
                 min_size=None,
                 max_size=None,
                 scale_factor=None,
                 sr_factor=None,
                 task=None,
                 img_shape=None,
                 mask_path=None,
                 scale_num=None):

        self.resampler = resampler

        # initial resizing if 'img_shape' is given
        self.img = self.load(img_path, img_shape)       # value range: [-1, 1]

        if task == 'photo':     # photo-realistic style transfer
            self.scale_factor = 1
            self.scale_num = 2
        elif task == 'sr' and sr_factor is not None and scale_num is not None:      # super resolution
            self.scale_num = scale_num
            self.scale_factor = 1 / (sr_factor ** (1 / (scale_num - 1)))
        elif scale_num is not None:
            # when 'scale_num' is explicitly given, use this and 'scale_factor' directly
            # usually used for inference
            self.scale_num = scale_num
            self.scale_factor = scale_factor
        else:
            # img will be resized using 'max_size' and 'min_size'
            # 'scale_factor' is re-calculated as closely as possible to the original
            # 'scale_num' is inferred
            self.scale_factor, self.scale_num, h, w = self.calc_scale_params(self.img, min_size, max_size, scale_factor)
            self.img = self.resampler(self.img, (h, w))

        self.img = self.img.cuda()

        if mask_path is not None:
            self.mask = self._gen_mask(mask_path, img_shape, task)
            self.mask = self.resampler(self.mask, self.img.shape[-2:])
        else:
            self.mask = None

        self.imgs = self._multiscale(self.img)

    def load(self, img_path, img_shape):
        img = Image.open(img_path).convert('RGB')
        img = ToTensor()(img)
        img = normalize(img)

        if img_shape is not None:
            return self.resampler(img.unsqueeze(0), img_shape)
        else:
            return img.unsqueeze(0)

    def _multiscale(self, img):
        sf = 1
        imgs = [img]
        *_, h, w = img.shape
        for scale in range(self.scale_num - 1):
            sf = sf * self.scale_factor
            img = self.resampler(img, (round(h * sf), round(w * sf)))
            imgs.append(img)

        return imgs[::-1]

    # Acknowledgement : This code block is a refactored version of that from official SinGAN Repo
    def _gen_mask(self, mask_path, img_shape, task):
        def _dilate(mask):
            if task == 'edit':
                element = morphology.disk(radius=7)     # edit
            elif task == 'harmo':
                element = morphology.disk(radius=20)    # harmo
            else:
                raise Exception

            mask = mask.squeeze(0).permute(1, 2, 0)
            mask = (mask / 2).clamp(-1, 1) * 255
            mask = mask.cpu().numpy()

            mask = mask.astype(np.uint8)
            mask = mask[:, :, 0]

            mask = morphology.binary_dilation(mask, selem=element)
            mask = filters.gaussian(mask, sigma=5)

            mask = color.rgb2gray(mask)
            mask = mask[:, :, None, None]
            mask = mask.transpose(3, 2, 0, 1)
            mask = torch.from_numpy(mask).type(torch.cuda.FloatTensor)
            mask = ((mask - 0.5) * 2).clamp(-1, 1)

            mask = mask.expand(1, 3, *mask.shape[-2:])
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            return mask

        mask = denormalize(self.load(mask_path, img_shape))
        mask = _dilate(mask)
        return mask

    def calc_scale_params(self, img, min_size, max_size, scale_factor):
        *_, h, w = img.shape
        long_side, short_side = max(h, w), min(h, w)

        min_factor = min_size / short_side
        max_factor = max_size / long_side

        long_side_max, short_side_max = round(long_side * max_factor), round(short_side * max_factor)
        long_side_min, short_side_min = round(long_side * min_factor), round(short_side * min_factor)

        scale_num = m.ceil(m.log(long_side_min / long_side_max, scale_factor)) + 1
        new_scale_factor = 1 / m.exp(m.log(long_side_max / long_side_min) * (1 / scale_num))

        if h > w:
            h, w = long_side_max, short_side_max
        else:
            w, h = long_side_max, short_side_max

        return new_scale_factor, scale_num + 1, h, w
