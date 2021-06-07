import time
import logging
from copy import deepcopy

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image

from utils.runner_utils import Loss, log_imgs, denormalize, log, shuffle_pixel
from networks.network import Network


class Runner:
    def __init__(self,
                 img_holder,
                 save_dir,
                 losses,
                 iter_per_scale,
                 pixel_shuffle_p,
                 img_ch,
                 net_ch,
                 lr,
                 betas,
                 scale_num,
                 resampler):

        self.img = img_holder.imgs

        self.save_dir = save_dir

        self.loss = Loss(losses)
        self.scale_num = scale_num

        self.iter_per_scale = iter_per_scale
        self.pixel_shuffle_p = pixel_shuffle_p

        self.net_list = [Network(img_ch, net_ch).cuda()]

        self.lr = lr
        self.betas = betas
        self.opt = Adam(self.net_list[-1].parameters(), lr, betas)
        self.opt_sch = CosineAnnealingLR(self.opt, self.iter_per_scale)

        self.resampler = resampler

    def save(self):
        torch.save({'net': [net.state_dict() for net in self.net_list]},
                   f"{self.save_dir}/model.torch")
        logging.info(f"Saved to {self.save_dir}/model.torch")

    def load(self):
        saved = torch.load(f"{self.save_dir}/model.torch")
        for _ in range(self.scale_num - 1):
            self._grow_network()
        for net, sn in zip(self.net_list, saved['net']):
            net.load_state_dict(sn)
        logging.info(f"Loaded from {self.save_dir}/model.torch")

    def _grow_network(self):
        self.net_list.append(deepcopy(self.net_list[-1]))
        self.net_list[-2].requires_grad_(False)
        self.opt = Adam(self.net_list[-1].parameters(), self.lr, self.betas)
        self.opt_sch = CosineAnnealingLR(self.opt, self.iter_per_scale)

    def _calc_loss(self, out, scale):
        return self.loss(out[scale], self.img[scale])

    def _step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.opt_sch.step()

    def _forward(self, scale_to, scale_from=0, infer=False):
        x = self.img[scale_from]
        out = []

        for scale in range(scale_from, scale_to + 1):
            net = self.net_list[scale]
            x = net(x)
            out.append(x)

            if not infer:
                with torch.no_grad():
                    x = shuffle_pixel(x, p=self.pixel_shuffle_p)

            if scale < scale_to:
                shape = self.img[scale + 1].shape[-2:]
                if x.shape[-2:] == shape:
                    x = x.detach()
                else:
                    with torch.no_grad():
                        x = self.resampler(x, shape)

        return out

    def train(self):
        start_time = time.time()
        start_time_inloop = time.time()

        for scale in range(self.scale_num):
            for iter_cnt in range(1, self.iter_per_scale + 1):
                out = self._forward(scale_to=scale, scale_from=0)
                loss = self._calc_loss(out, scale)
                self._step(loss)

                if iter_cnt % (self.iter_per_scale // 4) == 0:
                    log(self.save_dir, iter_cnt, self.iter_per_scale, out, start_time, start_time_inloop, scale, self.scale_num)
                    start_time_inloop = time.time()

            if scale < self.scale_num - 1:
                self._grow_network()

        self.save()

    def infer(self, save_dir_infer, mask, use_last_net_only=False):
        if use_last_net_only:
            self.net_list = [self.net_list[-1] for _ in self.net_list]

        log_imgs(save_dir_infer, [self.img[0]], 'inputs')

        with torch.no_grad():
            logging_imgs = []
            for i_from in range(self.scale_num):
                out = self._forward(scale_to=self.scale_num - 1, scale_from=i_from, infer=True)

                desc = [f'[{i_from + 1}|{self.scale_num}]']
                desc = '_'.join(desc)
                log_imgs(save_dir_infer, out[-1:], desc)

                if mask is not None:
                    out[-1] = (1 - mask) * self.img[-1] + mask * out[-1]
                    log_imgs(save_dir_infer, out[-1:], f"{desc}masked")

                logging_imgs.append(out[-1])

            grid = denormalize(torch.cat(logging_imgs, dim=0))
            save_image(grid, f"{save_dir_infer}/ALL.png", nrow=3)
