import argparse
import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from os import path as osp
from tqdm import tqdm
import ffmpeg
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import torch.onnx
import onnx
import onnxruntime as rt

from PIL import Image
import torchvision.transforms as transforms

model = rt.InferenceSession("RealESRGAN_x4plus1.onnx",providers=['CPUExecutionProvider'])
model_path=onnx.load("RealESRGAN_x4plus1.onnx")
print(onnx.helper.printable_graph(model_path.graph))
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
x = torch.randn(1, 3, 192, 192,requires_grad=True)
print(x)
# compute ONNX Runtime output prediction
rtinputs = {model.get_inputs()[0].name: to_numpy(x)}
rtoutput = model.run(None, rtinputs)
print(rtoutput)


img = Image.open("./inputs/0014.jpg")

resize = transforms.Resize([192, 192])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

rtinputs = {model.get_inputs()[0].name: to_numpy(img_y)}
rtoutput = model.run(None, rtinputs)
img_out_y = rtoutput[0]

# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)