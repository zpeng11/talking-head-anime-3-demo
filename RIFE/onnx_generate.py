from IFNet_HDv3_v4_25_lite import IFNet, Head
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.getcwd())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = os.path.join('RIFE', "flownet.pkl") 


dtype = torch.float if 'fp32' in sys.argv[1] else torch.half
num_interpo = int(sys.argv[2])
export_name = f'rife_x{num_interpo}_{"fp32" if dtype==torch.float else "fp16"}.onnx'
image_size = 512

image_slice = 256



def init_module(
    model_name: str,
    IFNet: nn.Module,
    scale: float,
    ensemble: bool,
    device: torch.device,
    dtype: torch.dtype,
    Head: nn.Module):
    state_dict = torch.load(model_name, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True, mmap=True)
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




class RIFEWrapped(nn.Module):
    def __init__(self, flownet, encoder):
        super().__init__()
        self.encoder = encoder
        self.flownet = flownet

        self.timesteps = torch.cat([torch.full([5, 1, image_slice, image_slice], float(1+i)/float(num_interpo), dtype=dtype, device=device) for i in range(num_interpo - 1)], dim=0)
        self.tenFlow_div = torch.tensor([(image_slice - 1.0) / 2.0, (image_slice - 1.0) / 2.0], dtype=torch.float, device=device)
        tenHorizontal = torch.linspace(-1.0, 1.0, image_slice, dtype=torch.float, device=device)
        tenHorizontal = tenHorizontal.view(1, 1, 1, image_slice).expand(-1, -1, image_slice, -1)
        tenVertical = torch.linspace(-1.0, 1.0, image_slice, dtype=torch.float, device=device)
        tenVertical = tenVertical.view(1, 1, image_slice, 1).expand(-1, -1, -1, image_slice)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)


    def forward(self, tha_img_0, tha_img_1):
        #input image with opencv style input, shape(image_size,image_size,4) range(0, 255), dtype uint8 BGRA layout
        if tha_img_0.dtype != torch.uint8 or tha_img_1.dtype != torch.uint8:
            raise ValueError('Data type error!')
        shapes = tha_img_0.shape
        if len(shapes) != 4 or shapes[1] != image_size or shapes[2] != image_size or shapes[3] != 4:
            raise ValueError('No a proper shape input')
        shapes = tha_img_1.shape
        if len(shapes) != 4 or shapes[1] != image_size or shapes[2] != image_size or shapes[3] != 4:
            raise ValueError('No a proper shape input')
        tha_img_0 = tha_img_0.to(dtype)
        tha_img_1 = tha_img_1.to(dtype)
        # BGRA to RGBA, uint8 to float/half, range to (0.0,1.0), shape to (1,4,image_size,image_size)
        img_0 = tha_img_0[:,:,:, [2,1,0,3]].permute(0, 3, 1, 2) / 255.0
        img_1 = tha_img_1[:,:,:, [2,1,0,3]].permute(0, 3, 1, 2) / 255.0

        img_0_slice_a = img_0[:,:3,0:256,128:128+256] 
        img_1_slice_a = img_1[:,:3,0:256,128:128+256]
        img_0_slice_b = img_0[:,:3,256:512,0:256] 
        img_1_slice_b = img_1[:,:3,256:512,0:256]
        img_0_slice_c = img_0[:,:3,256:512,256:512] 
        img_1_slice_c = img_1[:,:3,256:512,256:512]
        img_0_alpha_head = img_0[:,3,0:256,128:128+256].unsqueeze(0)
        img_1_alpha_head = img_1[:,3,0:256,128:128+256].unsqueeze(0)
        img_0_slice_d = torch.concat([img_0[:,1,0:256,128:128+256].unsqueeze(0), img_0_alpha_head, img_0[:,0,0:256,128:128+256].unsqueeze(0)], dim = 1) 
        img_1_slice_d = torch.concat([img_1[:,1,0:256,128:128+256].unsqueeze(0), img_1_alpha_head, img_1[:,0,0:256,128:128+256].unsqueeze(0)], dim = 1)

        img_0_slice_e = torch.concat([img_0[:,3,256:512,0:256].unsqueeze(0), img_0[:,3,256:512,256:512].unsqueeze(0), torch.zeros((1, 1, 256, 256), dtype=dtype, device=img_0.device)], dim = 1) 
        img_1_slice_e = torch.concat([img_1[:,3,256:512,0:256].unsqueeze(0), img_1[:,3,256:512,256:512].unsqueeze(0), torch.zeros((1, 1, 256, 256), dtype=dtype, device=img_1.device)], dim = 1)

        img_0_1_slices = torch.concat([img_0_slice_a, img_0_slice_b, img_0_slice_c, img_0_slice_d, img_0_slice_e, img_1_slice_a, img_1_slice_b, img_1_slice_c, img_1_slice_d, img_1_slice_e], dim=0)
        img_0_1_slices_encoded = self.encoder(img_0_1_slices)
        img_0_slices = []
        img_1_slices = []
        img_0_slices_encoded = []
        img_1_slices_encoded = []
        for i in range(num_interpo - 1):
            img_0_slices.append(img_0_1_slices[:5,:,:,:])
            img_1_slices.append(img_0_1_slices[5:10,:,:,:])
            img_0_slices_encoded.append(img_0_1_slices_encoded[:5,:,:,:])
            img_1_slices_encoded.append(img_0_1_slices_encoded[5:10,:,:,:])
        img_0_slices = torch.concat(img_0_slices, dim=0)
        img_1_slices = torch.concat(img_1_slices, dim=0)
        img_0_slices_encoded = torch.concat(img_0_slices_encoded, dim=0)
        img_1_slices_encoded = torch.concat(img_1_slices_encoded, dim=0)


        interpo_res = img_1.clone()

        rife_res = self.flownet(img_0_slices, 
                                img_1_slices, 
                                self.timesteps, 
                                self.tenFlow_div, 
                                self.backwarp_tenGrid, 
                                img_0_slices_encoded, 
                                img_1_slices_encoded)
        # return rife_res
        ret_res = []
        for i in range(num_interpo - 1):
            interpo_res[:,:3,0:256,128:128+256] = rife_res[0 + 5 * i ,:,:,:]
            interpo_res[:,:3,256:512,0:256] = rife_res[1 + 5 * i,:,:,:]
            interpo_res[:,:3,256:512,256:512] = rife_res[2 + 5 * i,:,:,:]
            interpo_res[:,3,0:256,128:128+256] = rife_res[3 + 5 * i,1,:,:]
            interpo_res[:,3,256:512,0:256] = rife_res[4 + 5 * i,0,:,:]
            interpo_res[:,3,256:512,256:512] = rife_res[4 + 5 * i,1,:,:]


            # CHW to HWC: (1,4,H,W) -> (H,W,4)
            res = interpo_res.permute(0, 2, 3, 1)  # (1,4,512,512) -> (1,512,512,4)
            res = res[:, :, :, [2,1,0,3]]  # RGBA back to BGRA
            res = (res * 255.0).clamp(0.0, 255.0)
            ret_res.append(res)

        #Append latest tha result
        ret_res.append(tha_img_1) #dtype back to uint8

        return torch.concat(ret_res, dim=0).to(torch.uint8)  #shape to (num_interpo, image_size, image_size, 4)
    
flownet, encoder = init_module(model_name, IFNet, 1.0, False, device, dtype, Head)
rife = RIFEWrapped(flownet, encoder).eval()





img_0 = (torch.rand((1, image_size,image_size, 4), device=device) * 255.0).to(torch.uint8)
img_1 = (torch.rand((1, image_size,image_size, 4), device=device) * 255.0).to(torch.uint8)
input_list = ['tha_img_0', 'tha_img_1']

output_list =["rife_outputs"]
input_tuple = (img_0, img_1)

import onnx
from onnxsim import simplify
torch.onnx.export(rife,
                  input_tuple,
                  export_name,
                  export_params=True,
                  opset_version=16,
                  do_constant_folding=True,
                  input_names= input_list,
                  output_names= output_list,
                  dynamo=False  # Add this line to disable the new exporter
                  )

onnx_model = onnx.load(export_name)
onnx.checker.check_model(onnx_model)
onnx_model_sim, check = simplify(onnx_model)
if check:
    onnx.save(onnx_model_sim, export_name)
else:
    raise ValueError("Simplify error")
