from argparse import Namespace

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision import transforms

from models.encoders import w_encoder


# 加载测试图片
def load_image(filename):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(filename)
    img = transform(img)
    return img.unsqueeze(dim=0)


# 开始压缩模型
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
    # 加载原本的模型
    ckpt = torch.load("../checkpoints/faces_w_encoder.pt", map_location='cpu')

    encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
    encoder.eval()
    encoder.to(device)
    with torch.no_grad():
        img = load_image('../data_content/unsplash-rDEOVtE7vOs.jpg').to(device)
        img = F.adaptive_avg_pool2d(img, 256)
        print("start save img ", img.shape)

        with torch.jit.optimized_execution(True):
            traced_script_module_encoder = torch.jit.trace(encoder, img, check_trace=True, optimize=True)

            path = "faces_w_encoder.jit"

            traced_script_module_encoder.save(path)
            script_model_vulkan = optimize_for_mobile(traced_script_module_encoder, backend='cpu')
            # 输出用于客户端的模型
            script_model_vulkan._save_for_lite_interpreter("./faces_w_encoder.ptl")


if __name__ == '__main__':
    print(torch.__version__)

    main()
