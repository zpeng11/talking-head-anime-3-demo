import sys
import os
sys.path.append(os.getcwd())
from trt.trt_utils import *
from os.path import join
import torch
import numpy as np

import pycuda.driver as cuda
cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()
from time import time


class Poser:
    def __init__(self, model:str) -> None:
        model_type = 'seperable' if 'separable' in model else 'standard'
        dtype = 'fp16' if 'half' in model else 'fp32'
        self.dtype = torch.half if 'half' in model else torch.float
        model_dir = join('rt_poser', model_type, dtype)
        self.combiner = Processor(load_engine(join(model_dir, 'combiner'+'.trt')), 4)
        self.decomposer = Processor(load_engine(join(model_dir, 'decomposer'+'.trt')), 1)
        self.editor = Processor(load_engine(join(model_dir, 'editor'+'.trt')), 4)
        self.morpher = Processor(load_engine(join(model_dir, 'morpher'+'.trt')), 4)
        self.rotator = Processor(load_engine(join(model_dir, 'rotator'+'.trt')), 2)

    def get_image_size(self):
        return 512
    
    def get_dtype(self):
        return self.dtype

    def cal(self, img, pose):
        cuda_driver_context.push()
        decomposer_res = self.decomposer.inference([img])
        combiner_res = self.combiner.inference([img, decomposer_res[0], decomposer_res[1], pose[:, :12]])
        morpher_res = self.morpher.inference([img, combiner_res[0], pose[:,12:12+27], combiner_res[1]])
        rotator_res = self.rotator.inference([morpher_res[1], pose[:,12+27:]])
        editor_res = self.editor.inference([morpher_res[0], rotator_res[0], rotator_res[1], pose[:,12+27:]])
        cuda_driver_context.pop()
        return editor_res[0]

    def pose(self, pt_img:torch.Tensor, pt_pose:torch.Tensor) -> torch.Tensor:
        t1 = time()
        res = self.cal(pt_img.cpu().detach().numpy().reshape(1,4,512,512), pt_pose.cpu().detach().numpy().reshape(1,45))
        print(time() - t1)
        return torch.from_numpy(res).to('cuda:0')
        # return pt_img.reshape(1,4,512,512)
