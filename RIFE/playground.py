from IFNet_HDv3_v4_25_lite import IFNet, Head
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from trt_utils import *
from torchvision.transforms.functional import adjust_sharpness

dtype = torch.half
modulo = 128

device = torch.device("cuda", 0)
model_name = "flownet.pkl"
export_name = './export/rife_256'

image_size = 512
internal_size = 256
sharpen_factor = 1

scale = 1.0
ensemble = False


def init_module(
    model_name: str,
    IFNet: nn.Module,
    scale: float,
    ensemble: bool,
    device: torch.device,
    dtype: torch.dtype,
    Head: nn.Module):
    state_dict = torch.load(model_name, map_location="cpu", weights_only=True, mmap=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    with torch.device("meta"):
        flownet = IFNet(scale, ensemble)
    flownet.load_state_dict(state_dict, strict=False, assign=True)
    flownet.eval().to(device, dtype)

    if Head is not None:
        encode_state_dict = {k.replace("encode.", ""): v for k, v in state_dict.items() if "encode." in k}

        if isinstance(Head, nn.Sequential):
            encode = Head
        else:
            with torch.device("meta"):
                encode = Head()
        encode.load_state_dict(encode_state_dict, assign=True)
        encode.eval().to(device, dtype)

        return flownet, encode

    return flownet, None



pw = image_size
ph = image_size
timestep = torch.full([1, 1, ph, pw], 0.5, dtype=dtype, device=device)
tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=device)
tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=device)
tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=device)
tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)



class RIFEWrapped(nn.Module):
    def __init__(self, timestep, tenFlow_div, backwarp_tenGrid, flownet, encoder):
        super().__init__()
        self.encoder = encoder
        self.flownet = flownet

        self.timestep = torch.full([1, 1, internal_size, internal_size], 0.5, dtype=dtype, device=device)
        self.tenFlow_div = torch.tensor([(internal_size - 1.0) / 2.0, (internal_size - 1.0) / 2.0], dtype=torch.float, device=device)
        tenHorizontal = torch.linspace(-1.0, 1.0, internal_size, dtype=torch.float, device=device)
        tenHorizontal = tenHorizontal.view(1, 1, 1, internal_size).expand(-1, -1, internal_size, -1)
        tenVertical = torch.linspace(-1.0, 1.0, internal_size, dtype=torch.float, device=device)
        tenVertical = tenVertical.view(1, 1, internal_size, 1).expand(-1, -1, -1, internal_size)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)


    def forward(self, img_0, img_1):

        if internal_size != image_size:
            img_0 = F.interpolate(img_0, size=(internal_size, internal_size), mode='bilinear', align_corners=False)
            img_1 = F.interpolate(img_1, size=(internal_size, internal_size), mode='bilinear', align_corners=False)

        encoded_0 = self.encoder(img_0)
        encoded_1 = self.encoder(img_1)

        res = self.flownet(img_0, img_1, self.timestep, self.tenFlow_div, self.backwarp_tenGrid, encoded_0, encoded_1)

        if internal_size != image_size:
            res = F.interpolate(res, size=(image_size, image_size), mode='bilinear', align_corners=False)
            # res = adjust_sharpness(res, 1)
        return res 
    
flownet, encoder = init_module(model_name, IFNet, scale, ensemble, device, dtype, Head)
rife = RIFEWrapped(timestep, tenFlow_div, backwarp_tenGrid, flownet, encoder).eval()




for i in range(10):
    img_0 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)
    img_1 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)

    ec0 = encoder(img_0)
    ec1 = encoder(img_1)
    
    ref_output = flownet(img_0, img_1, timestep, tenFlow_div, backwarp_tenGrid, ec0, ec1).cpu().detach().numpy()
    wrapped_output  = rife(img_0, img_1).cpu().detach().numpy()
    
    print(wrapped_output.mean(), wrapped_output.max(), ref_output.mean(), ref_output.max())
    print("MSE is: ",((ref_output - wrapped_output) ** 2).mean())


img_0 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)
img_1 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)
input_list = ['img_0', 'img_1']
output_list = ['result']
input_tuple = (img_0, img_1)

import onnx
from onnxsim import simplify
torch.onnx.export(rife,
                  input_tuple,
                  export_name+".onnx",
                  export_params=True,
                  opset_version=16,
                  do_constant_folding=True,
                  input_names= input_list,
                  output_names= output_list
                  )

onnx_model = onnx.load(export_name+".onnx")
onnx.checker.check_model(onnx_model)
onnx_model_sim, check = simplify(onnx_model)
if check:
    onnx.save(onnx_model_sim, export_name+".onnx")
else:
    raise ValueError("Simplify error")


convert('.','.',export_name, 'fp16')


rife_engine = load_engine(export_name+'.trt')
rife_proc = Processor(rife_engine, 2)

for i in range(20):
    img_0 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)
    img_1 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)
    ec0 = encoder(img_0)
    ec1 = encoder(img_1)

    wrapped_output  = flownet(img_0, img_1, timestep, tenFlow_div, backwarp_tenGrid, ec0, ec1).cpu().detach().numpy()

    trt_res = rife_proc.inference([img_0.cpu().detach().numpy(), img_1.cpu().detach().numpy()])[0]
    
    # print(wrapped_output.mean(), wrapped_output.max(), trt_res.mean(), trt_res.max())
    print("MSE is: ",((trt_res - wrapped_output) ** 2).mean())


from tqdm import tqdm
from time import time 
t1 = time()
for i in tqdm(range(1000)):
    rife_proc.kickoff()

print(time() - t1)