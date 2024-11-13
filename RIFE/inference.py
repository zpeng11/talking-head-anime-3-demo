from trt_utils import *
import onnxruntime
import torch


export_name = './export/rife_512'
dtype = torch.half
device = torch.device("cuda", 0)
image_size = 512
