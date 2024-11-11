import subprocess
import os
from os.path import join
import sys
from pathlib import Path
sys.path.insert(0, join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy
import onnxruntime as ort
import os
from onnxruntime.quantization import CalibrationDataReader,QuantFormat,quantize_static,QuantType,CalibrationMethod
from PIL import Image
import torch
from torch.nn.functional import interpolate
from tha3.util import resize_PIL_image,extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image
import tha3
from tqdm import tqdm


MODEL_NAME = "separable_float"
MODEL_BEFORE = './onnx_model/fp32/editor.onnx'
PREPROCESSED_MODEL = './quantize/editor_infr.onnx'
POSE_QUANTIZE_MODEL = './quantize/editor_quant.onnx'
IMAGES_DIR = './data/images'
NUM_OF_SETUPS = 5
DEVICE_NAME = 'cuda:0'
providers = [("CUDAExecutionProvider", {"device_id": 0, #torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
sess_options = ort.SessionOptions()
sess_options.log_severity_level=1

device = torch.device(DEVICE_NAME)
dtype = torch.float32




p = subprocess.Popen('python -m onnxruntime.quantization.preprocess --input '+ MODEL_BEFORE + ' --output ' + PREPROCESSED_MODEL, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in p.stdout.readlines():
    print(str(line))

def load_poser(model: str, device: torch.device):
    print("Using the %s model." % model)
    if model == "standard_float":
        from tha3.poser.modes.standard_float import create_poser
        return create_poser(device)
    elif model == "standard_half":
        from tha3.poser.modes.standard_half import create_poser
        return create_poser(device)
    elif model == "separable_float":
        from tha3.poser.modes.separable_float import create_poser
        return create_poser(device)
    elif model == "separable_half":
        from tha3.poser.modes.separable_half import create_poser
        return create_poser(device)
    else:
        raise RuntimeError("Invalid model: '%s'" % model)

poser = load_poser(MODEL_NAME, DEVICE_NAME)
eyebrow_decomposer = poser.get_modules()['eyebrow_decomposer']
eyebrow_morphing_combiner = poser.get_modules()['eyebrow_morphing_combiner']
face_morpher = poser.get_modules()['face_morpher']
two_algo_face_body_rotator = poser.get_modules()['two_algo_face_body_rotator']
editor = poser.get_modules()['editor']

eyebrow_pose = torch.zeros((1, 12), device=device, dtype=dtype)
face_pose = torch.zeros((1,27), device=device, dtype=dtype)
rotation_poses = [(torch.rand((1,6), device=device, dtype=dtype) * 2.0 - 1.0) for _ in range(NUM_OF_SETUPS)]

def calcualteUntilEditor(im):
    im_decomposer_crop = im[:,:, 64:192, 64 + 128:192 + 128].clone()

    eyebrow_decomposer_torch_res = eyebrow_decomposer(im_decomposer_crop)
    eyebrow_morphing_combiner_torch_res = eyebrow_morphing_combiner(eyebrow_decomposer_torch_res[3], eyebrow_decomposer_torch_res[0], eyebrow_pose)

    im_morpher_crop = im[:, :, 32:32 + 192, (32 + 128):(32 + 192 + 128)].clone()
    im_morpher_crop[:, :, 32:32 + 128, 32:32 + 128] = eyebrow_morphing_combiner_torch_res[2]
    face_morpher_torch_res = face_morpher(im_morpher_crop, face_pose)

    face_morphed_full = im.clone()
    face_morphed_full[:, :, 32:32 + 192, 32 + 128:32 + 192 + 128] = face_morpher_torch_res[0]
    face_morphed_half = interpolate(face_morphed_full, size=(256, 256), mode='bilinear', align_corners=False)
    
    results = []

    for rotation_pose in rotation_poses:
        two_algo_face_body_rotator_torch_res = two_algo_face_body_rotator(face_morphed_half, rotation_pose)

        input_original_image = face_morphed_full
        half_warped_image = two_algo_face_body_rotator_torch_res[1]
        full_warped_image = interpolate(half_warped_image, size=(512, 512), mode='bilinear', align_corners=False)
        half_grid_change = two_algo_face_body_rotator_torch_res[2]
        full_grid_change = interpolate(half_grid_change, size=(512, 512), mode='bilinear', align_corners=False)
        results.append({
            'morphed_image':input_original_image.cpu().detach().numpy(),
            'rotated_warped_image':full_warped_image.cpu().detach().numpy(),
            'rotated_grid_change':full_grid_change.cpu().detach().numpy(),
            'rotation_pose':rotation_pose.cpu().detach().numpy()
            })
    return results

def processImages(dir):
    all_inputs = []
    directory = os.fsencode(dir)
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".png"): 
            pil_image = resize_PIL_image(extract_PIL_image_from_filelike('data/images/'+filename), size=(512,512))
            torch_input_image = extract_pytorch_image_from_PIL_image(pil_image).reshape(1,4,512,512).to(device)
            all_inputs.extend(calcualteUntilEditor(torch_input_image))
    return all_inputs
# res = processImages(IMAGES_DIR)
# print(len(res))
# print(res[0])
# for re in res:
#     for k,v in re.items():
#         print(k, v.shape)

class PoserDataReader(CalibrationDataReader):
    def __init__(self, img_path: str):
        self.enum_data = None
        self.data_list = processImages(img_path)
    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                self.data_list
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

dr = PoserDataReader(IMAGES_DIR)
quantize_static(
        PREPROCESSED_MODEL,
        POSE_QUANTIZE_MODEL,
        dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=['Conv'],
        calibrate_method=CalibrationMethod.Entropy,
        weight_type=QuantType.QInt8,
        extra_options={
            'ActivationSymmetric':True
        }
    )



import onnx
onnx_model_fp32 = onnx.load(MODEL_BEFORE)
onnx_model_quant = onnx.load(POSE_QUANTIZE_MODEL)
onnx.checker.check_model(onnx_model_quant)


import onnxruntime as ort
import numpy as np
ort_sess_fp = ort.InferenceSession(MODEL_BEFORE, sess_options=sess_options, providers=providers)
ort_sess_quant = ort.InferenceSession(POSE_QUANTIZE_MODEL, sess_options=sess_options, providers=providers)
for data in dr.data_list:
    ort_sess_fp_res = ort_sess_fp.run(None,data)
    ort_sess_quant_res = ort_sess_quant.run(None,data)
    print("MSE is: ",((ort_sess_fp_res[0] - ort_sess_quant_res[0]) ** 2).mean())
    

for i in tqdm(range(100)):
    ort_sess_fp.run(None,dr.data_list[0])

for i in tqdm(range(100)):
    ort_sess_quant.run(None,dr.data_list[0])
