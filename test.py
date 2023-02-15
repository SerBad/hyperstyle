import hiddenlayer as h
from models.encoders import w_encoder

import torch

encode = w_encoder.WEncoder()
x = torch.randn(1,1, 18, 512)
vis_graph = h.build_graph(encode, x)  # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
vis_graph.save("./demo1.png")  # 保存图像的路径
