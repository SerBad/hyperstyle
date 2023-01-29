import math
import torch
from torch import nn
import copy
from argparse import Namespace
from torchvision import transforms
from models.encoders import w_encoder
from PIL import Image
from models.encoders.psp import pSp
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
from models.hypernetworks.hypernetwork import SharedWeightsHyperNetResNet, SharedWeightsHyperNetResNetSeparable
from utils.resnet_mapping import RESNET_MAPPING
from torch.nn import functional as F
import netron


def load_image(filename):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(filename)
    img = transform(img)
    return img.unsqueeze(dim=0)


def main():
    def get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt

    device = "cpu"
    opts = Namespace()
    opts.output_size = 1024
    encoder = w_encoder.WEncoder(50, 'ir_se', opts)
    ckpt = torch.load("../checkpoints/faces_w_encoder.pt", map_location='cpu')

    encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
    encoder.eval()
    encoder.to(device)
    with torch.no_grad():
        img = load_image('../data_content/unsplash-rDEOVtE7vOs.jpg').to(device)
        img = F.adaptive_avg_pool2d(img, 256)
        print("start save img ", img.shape)
        traced_script_module_encoder = torch.jit.trace(encoder, img, check_trace=True, optimize=True)
        path = ".faces_w_encoder.jit"
        traced_script_module_encoder.save(".faces_w_encoder.jit")

        netron.start(path)


if __name__ == '__main__':
    main()
