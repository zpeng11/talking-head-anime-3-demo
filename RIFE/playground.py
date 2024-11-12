from IFNet_HDv3_v4_25_lite import IFNet, Head
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from trt_utils import *

dtype = torch.half
modulo = 128

device = torch.device("cuda", 0)
model_name = "flownet.pkl"

image_size = 384
w = image_size
h = image_size

scale = 1.0
ensemble = False
pw = w
ph = h
padding = (0,0,0,0)
need_pad = False

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

flownet, encode = init_module(model_name, IFNet, scale, ensemble, device, dtype, Head)

flownet = flownet.eval()
encode = encode.eval()


factor_num = 2
factor_den = 1
timestep = torch.full([1, 1, ph, pw], 0.5, dtype=dtype, device=device)


tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=device)
tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=device)
tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=device)
tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)


img_0 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)
img_1 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)
encode_0 = encode(img_0)
encode_1 = encode(img_1)

output = flownet(img_0, img_1, timestep, tenFlow_div, backwarp_tenGrid, encode_0, encode_1).cpu().detach().numpy()
print(output.shape)

class RIFEWrapped(nn.Module):
    def __init__(self, timestep=timestep, tenFlow_div=tenFlow_div, backwarp_tenGrid=backwarp_tenGrid, head =encode, flownet = flownet):
        super().__init__()
        self.head = head
        self.flownet = flownet
        self.timestep=timestep
        self.tenFlow_div=tenFlow_div
        self.backwarp_tenGrid=backwarp_tenGrid
    def forward(self, img_0, img_1):
        # half_0 = F.interpolate(img_0, size=(256, 256), mode="bilinear", align_corners=False)
        # half_1 = F.interpolate(img_1, size=(256, 256), mode="bilinear", align_corners=False)
        encoded_0 = encode(img_0)
        encoded_1 = encode(img_1)
        res = flownet(img_0, img_0, self.timestep, self.tenFlow_div, self.backwarp_tenGrid, encoded_0, encoded_1)
        return res #F.interpolate(res, size=(512, 512), mode="bilinear", align_corners=False)
    
rife = RIFEWrapped().eval()
img_0 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)
img_1 = torch.rand((1,3,image_size,image_size), dtype=dtype, device=device)
res  = rife(img_0, img_1)
print(res.shape)

input_list = ['img_0', 'img_1']
output_list = ['result']
input_tuple = (img_0, img_1)




import onnx
from onnxsim import simplify
torch.onnx.export(rife,
                  input_tuple,
                  "rife.onnx",
                  export_params=True,
                  opset_version=16,
                  do_constant_folding=True,
                  input_names= input_list,
                  output_names= output_list
                  )

onnx_model = onnx.load("rife.onnx")
onnx.checker.check_model(onnx_model)
onnx_model_sim, check = simplify(onnx_model)
if check:
    onnx.save(onnx_model_sim, "rife.onnx")
else:
    raise ValueError("Simplify error")


convert('.','.','rife', 'fp16')


rife_engine = load_engine('rife.trt')
rife_proc = Processor(rife_engine, 2)

from tqdm import tqdm
for i in tqdm(range(1000000)):
    rife_proc.kickoff()