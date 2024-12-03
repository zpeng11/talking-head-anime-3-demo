from IFNet_HDv3_v4_25_lite import IFNet, Head
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from trt_utils import *
import sys
import os
import cv2
sys.path.append(os.getcwd())

device = torch.device("cuda", 0)
model_name = os.path.join('RIFE', "flownet.pkl") 
image_size = 512

dtype = torch.float if 'fp32' in sys.argv[1] else torch.half
num_interpo = int(sys.argv[2])
export_name = sys.argv[3]



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
timesteps = [torch.full([1, 1, ph, pw], float(1+i)/float(num_interpo), dtype=dtype, device=device) for i in range(num_interpo - 1)]
timestep1 = torch.full([1, 1, ph, pw], 1.0/6.0, dtype=dtype, device=device)
timestep2 = torch.full([1, 1, ph, pw], 2.0/6.0, dtype=dtype, device=device)
timestep3 = torch.full([1, 1, ph, pw], 3.0/6.0, dtype=dtype, device=device)
timestep4 = torch.full([1, 1, ph, pw], 4.0/6.0, dtype=dtype, device=device)
timestep5 = torch.full([1, 1, ph, pw], 5.0/6.0, dtype=dtype, device=device)
tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=device)
tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=device)
tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=device)
tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

def torch_linear_to_srgb(x):
    x = torch.clip(x, 0.0, 1.0)
    return torch.where(torch.le(x, 0.003130804953560372), x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

class RIFEWrapped(nn.Module):
    def __init__(self, flownet, encoder):
        super().__init__()
        self.encoder = encoder
        self.flownet = flownet

        self.timesteps = [torch.full([1, 1, ph, pw], float(1+i)/float(num_interpo), dtype=dtype, device=device) for i in range(num_interpo - 1)]
        self.tenFlow_div = torch.tensor([(image_size - 1.0) / 2.0, (image_size - 1.0) / 2.0], dtype=torch.float, device=device)
        tenHorizontal = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float, device=device)
        tenHorizontal = tenHorizontal.view(1, 1, 1, image_size).expand(-1, -1, image_size, -1)
        tenVertical = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float, device=device)
        tenVertical = tenVertical.view(1, 1, image_size, 1).expand(-1, -1, -1, image_size)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)


    def forward(self, tha_img_0, tha_img_1):
        #input image with opencv style input, shape(512,512,4) range(0, 255), dtype uint8 BGRA layout
        if tha_img_0.dtype != torch.uint8 or tha_img_1.dtype != torch.uint8:
            raise ValueError('Data type error!')
        shapes = tha_img_0.shape
        if len(shapes) != 3 or shapes[0] != 512 or shapes[1] != 512 or shapes[2] != 4:
            raise ValueError('No a proper shape input')
        shapes = tha_img_1.shape
        if len(shapes) != 3 or shapes[0] != 512 or shapes[1] != 512 or shapes[2] != 4:
            raise ValueError('No a proper shape input')

        # BGRA to RGBA, uint8 to float/half, range to (0.0,1.0), shape to (1,4,512,512)
        img_0 = (tha_img_0.to(dtype)[:,:, [2,1,0,3]] / 255.0).reshape(512 * 512, 4).transpose(0,1).reshape(1,4, 512,512)
        img_1 = (tha_img_1.to(dtype)[:,:, [2,1,0,3]] / 255.0).reshape(512 * 512, 4).transpose(0,1).reshape(1,4, 512,512)

        interpo_res = [torch.zeros((1,4,512, 512), dtype=dtype, device = device) for i in range(num_interpo - 1)]

        for i in range(len(interpo_res)):
            if i == 0:
                interpo_res[i][:, 3, :, :] = img_0[:,3,:,:]
            else:
                interpo_res[i][:, 3, :, :] = img_1[:,3,:,:]


        encoded_0 = self.encoder(img_0[:, :3, :, :])
        encoded_1 = self.encoder(img_1[:, :3, :, :])

        ret_res = []

        for i in range(num_interpo - 1):
            interpo_res[i][:, :3, :, :] = self.flownet(img_0[:, :3, :, :], 
                                                        img_1[:, :3, :, :], 
                                                        self.timesteps[i], 
                                                        self.tenFlow_div, 
                                                        self.backwarp_tenGrid, 
                                                        encoded_0, 
                                                        encoded_1)
            
            res = interpo_res[i].reshape(4, 512 * 512).transpose(0, 1).reshape(512, 512, 4) #Reshape back to (512, 512, 4)
            res = res[:, :, [2,1,0,3]] #RGBA back to BGRA
            res = torch.clip(res * 255.0, 0.0, 255.0) #range back to (0.0, 255.0)
            ret_res.append(res.to(torch.uint8)) #dtype back to uint8

        #Append latest tha result
        ret_res.append(tha_img_1)

        return ret_res
    
flownet, encoder = init_module(model_name, IFNet, 1.0, False, device, dtype, Head)
rife = RIFEWrapped(flownet, encoder).eval()





img_0 = (torch.rand((image_size,image_size, 4), device=device) * 255.0).to(torch.uint8)
img_1 = (torch.rand((image_size,image_size, 4), device=device) * 255.0).to(torch.uint8)
input_list = ['tha_img_0', 'tha_img_1']

output_list =[]
for i in range(num_interpo - 1):
    output_list.append(f'interpo_{i}')
output_list.append('tha_res')
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



# convert('.','.',export_name, 'fp16')


# rife_engine = load_engine(export_name+'.trt')
# rife_proc = Processor(rife_engine, 2)

# for i in range(20):
#     img_0 = torch.rand((1,4,image_size,image_size), dtype=dtype, device=device)
#     img_1 = torch.rand((1,4,image_size,image_size), dtype=dtype, device=device)
#     ec0 = encoder(img_0[:,:3,:,:])
#     ec1 = encoder(img_1[:,:3,:,:])

#     ref_output_1  = flownet(img_0[:,:3,:,:], img_1[:,:3,:,:], timestep1, tenFlow_div, backwarp_tenGrid, ec0, ec1).cpu().detach().numpy()
#     ref_output_2  = flownet(img_0[:,:3,:,:], img_1[:,:3,:,:], timestep2, tenFlow_div, backwarp_tenGrid, ec0, ec1).cpu().detach().numpy()
#     ref_output_3  = flownet(img_0[:,:3,:,:], img_1[:,:3,:,:], timestep3, tenFlow_div, backwarp_tenGrid, ec0, ec1).cpu().detach().numpy()

#     trt_res = rife_proc.inference([img_0.cpu().detach().numpy(), img_1.cpu().detach().numpy()])
    
#     # print(wrapped_output.mean(), wrapped_output.max(), trt_res.mean(), trt_res.max())
#     print("1st MSE is: ",((trt_res[0][:,:3,:,:] - ref_output_1) ** 2).mean())
#     print("1st MSE is: ",((trt_res[1][:,:3,:,:] - ref_output_2) ** 2).mean())
#     print("1st MSE is: ",((trt_res[2][:,:3,:,:] - ref_output_3) ** 2).mean())


# from tqdm import tqdm
# from time import time 
# t1 = time()
# for i in tqdm(range(1000)):
#     rife_proc.kickoff()

# print(time() - t1)