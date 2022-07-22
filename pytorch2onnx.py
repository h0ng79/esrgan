import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import onnxruntime as rt
from PIL import Image
import torchvision.transforms as transforms



def main(args):
    # An instance of the model

    # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    if args.params:
        keyname = 'params'
    else:
        keyname = 'params_ema'
    model.load_state_dict(torch.load((args.input),map_location=torch.device('cpu'))[keyname])
    # model.load_state_dict(torch.load(args.input,map_location=torch.device('cpu')))
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    # An example input
    x = torch.rand(1, 3, 64, 64)
    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, args.output, opset_version=11, export_params=True)

    model.load_state_dict(torch.load((args.input),map_location=torch.device('cpu'))[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()
    # An example input

    x = torch.randn(1, 3, 192, 192)
    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx._export(model,x, args.output, opset_version=11, export_params=True)

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = "optimized_model.onnx"
    session = rt.InferenceSession("animev3.onnx", sess_options)
    print(torch_out.shape)


if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()

    # parser.add_argument('--input_data',type=str,default='inputs/00003.png',help='Input_image')
    parser.add_argument(
        '--input', type=str, default='experiments/pretrained_models/realesr-animevideov3.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='animev3.onnx', help='Output onnx path')
    parser.add_argument('--params', action='store_false',help='Use params instead of params_ema')
    # parser.add_argument("--ouput_data",type=str,default='results/0003.png',help='output_data')
    args = parser.parse_args()

    main(args)
