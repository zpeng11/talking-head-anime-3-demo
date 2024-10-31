import subprocess
import os
os.chdir('..')
import sys
sys.path.insert(1, './')
import numpy
import onnxruntime
import os
from onnxruntime.quantization import CalibrationDataReader,QuantFormat,quantize_static,QuantType
from PIL import Image
import torch
from torch.nn.functional import interpolate
from tha3.util import resize_PIL_image,extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image
import tha3
from tqdm import tqdm

MODEL_BEFORE = './editor.onnx'
PREPROCESSED_MODEL = './quantize/sim_editor_infr.onnx'
POSE_QUANTIZE_MODEL = './quantize/sim_editor_quant.onnx'
IMAGES_DIR = './data/images'
NUM_OF_SETUPS = 5





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

poser = load_poser('standard_float', 'cpu')
eyebrow_decomposer = poser.get_modules()['eyebrow_decomposer']
eyebrow_morphing_combiner = poser.get_modules()['eyebrow_morphing_combiner']
face_morpher = poser.get_modules()['face_morpher']
two_algo_face_body_rotator = poser.get_modules()['two_algo_face_body_rotator']
editor = poser.get_modules()['editor']

eyebrow_pose = torch.zeros((1, 12))
face_pose = torch.zeros((1,27))
rotation_poses = [(torch.rand((1,6)) * 2.0 - 1.0) for _ in range(NUM_OF_SETUPS)]

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

    for rotation_pose in tqdm(rotation_poses):
        two_algo_face_body_rotator_torch_res = two_algo_face_body_rotator(face_morphed_half, rotation_pose)

        input_original_image = face_morphed_full
        half_warped_image = two_algo_face_body_rotator_torch_res[1]
        full_warped_image = interpolate(half_warped_image, size=(512, 512), mode='bilinear', align_corners=False)
        half_grid_change = two_algo_face_body_rotator_torch_res[2]
        full_grid_change = interpolate(half_grid_change, size=(512, 512), mode='bilinear', align_corners=False)
        results.append({
            'input_original_image':input_original_image.detach().numpy(),
            'full_warped_image':full_warped_image.detach().numpy(),
            'full_grid_change':full_grid_change.detach().numpy(),
            'rotation_pose':rotation_pose.detach().numpy()
            })
    return results

def processImages(dir):
    all_inputs = []
    directory = os.fsencode(dir)
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".png"): 
            pil_image = resize_PIL_image(extract_PIL_image_from_filelike('data/images/'+filename), size=(512,512))
            torch_input_image = extract_pytorch_image_from_PIL_image(pil_image).reshape(1,4,512,512)
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
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        per_channel=False,
        weight_type=QuantType.QUInt8,
    )



import onnx
onnx_model_fp32 = onnx.load(MODEL_BEFORE)
onnx_model_quant = onnx.load(POSE_QUANTIZE_MODEL)
onnx.checker.check_model(onnx_model_quant)


import onnxruntime as ort
import numpy as np
ort_sess_fp = ort.InferenceSession(MODEL_BEFORE)
ort_sess_quant = ort.InferenceSession(POSE_QUANTIZE_MODEL)
for data in dr.data_list:
    ort_sess_fp_res = ort_sess_fp.run([            
                'output_color_changed',
                'output_color_change_alpha',
                'output_color_change',
                'output_warped_image',
                'output_grid_change',
                ],data)
    ort_sess_quant_res = ort_sess_quant.run([            
                'output_color_changed',
                'output_color_change_alpha',
                'output_color_change',
                'output_warped_image',
                'output_grid_change',
                ],data)
    print("MSE is: ",((ort_sess_fp_res[0] - ort_sess_quant_res[0]) ** 2).mean())
    
from time import time
t1 = time()
for i in range(10):
    ort_sess_fp.run([            
                'output_color_changed',
                'output_color_change_alpha',
                'output_color_change',
                'output_warped_image',
                'output_grid_change',
                ],dr.data_list[0])
print(time() - t1)
t1 = time()
for i in range(10):
    ort_sess_quant.run([            
                'output_color_changed',
                'output_color_change_alpha',
                'output_color_change',
                'output_warped_image',
                'output_grid_change',
                ],dr.data_list[0])
print(time() - t1)