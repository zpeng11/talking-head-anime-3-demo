import torch
import numpy
import PIL.Image
from tha3.util import resize_PIL_image,extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image

MODEL_NAME = "standard_float"
DEVICE_NAME = 'cpu'
IMAGE_INPUT = "data\images\crypko_03.png"
ONNX_MODEL_NAME = "face_morpher.onnx"


device = torch.device(DEVICE_NAME)


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
pose_size = poser.get_num_parameters()

eyebrow_decomposer = poser.get_modules()['eyebrow_decomposer']
eyebrow_morphing_combiner = poser.get_modules()['eyebrow_morphing_combiner']
face_morpher = poser.get_modules()['face_morpher']

def load_input_img():
    pil_image = resize_PIL_image(extract_PIL_image_from_filelike(IMAGE_INPUT), size=(512,512))
    return extract_pytorch_image_from_PIL_image(pil_image).to(device)

im = load_input_img().reshape(1,4,512,512)
# im = torch.rand(1, 4, 512, 512) * 2.0 - 1.0
pose = torch.zeros(1, pose_size, dtype=poser.get_dtype())
poser.pose(im, pose)

im_decomposer_crop = im[:,:, 64:192, 64 + 128:192 + 128].clone()

eyebrow_pose = torch.zeros((1, 12))

eyebrow_decomposer_torch_res = eyebrow_decomposer(im_decomposer_crop)

eyebrow_morphing_combiner_torch_res = eyebrow_morphing_combiner(eyebrow_decomposer_torch_res[3], eyebrow_decomposer_torch_res[0], eyebrow_pose)

im_morpher_crop = im[:, :, 32:32 + 192, (32 + 128):(32 + 192 + 128)].clone()
im_morpher_crop[:, :, 32:32 + 128, 32:32 + 128] = eyebrow_morphing_combiner_torch_res[2]
face_pose = torch.zeros((1,27))
face_morpher_torch_res = face_morpher(im_morpher_crop, face_pose)
torch.onnx.export(face_morpher,               # model being run
                  (im_morpher_crop, face_pose),                         # model input (or a tuple for multiple inputs)
                  ONNX_MODEL_NAME,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['eyebrow_image_no_combine_alpha', 'face_pose'],   # the model's input names
                  output_names = [  
            'output_image', #0
            'eye_alpha', #1
            'eye_color_change', #2
            'iris_mouth_image_1', #3
            'iris_mouth_alpha', #4
            'iris_mouth_color_change', #5
            'iris_mouth_image_0', #6
            ]) 

import onnx
onnx_model = onnx.load(ONNX_MODEL_NAME)
onnx.checker.check_model(onnx_model)
from onnxsim import simplify
onnx_model_sim, check = simplify(onnx_model)
assert check,"Simply is not avaialable"
onnx.save(onnx_model_sim,"sim_"+ONNX_MODEL_NAME)

import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession("sim_"+ONNX_MODEL_NAME)
face_morpher_onnx_res = ort_sess.run([            
            'output_image', #0
            'eye_alpha', #1
            'eye_color_change', #2
            'iris_mouth_image_1', #3
            'iris_mouth_alpha', #4
            'iris_mouth_color_change', #5
            'iris_mouth_image_0', #6
            ],
            {'eyebrow_image_no_combine_alpha': im_morpher_crop.detach().numpy(),
             "face_pose":face_pose.detach().numpy()
             })


# from PIL import Image
# INPUT_SIZE = 128
for i in range(len(face_morpher_onnx_res)):
    print("MSE is: ",((face_morpher_onnx_res[i] - face_morpher_torch_res[i].detach().numpy()) ** 2).mean())
#     print(eyebrow_morphing_combiner_onnx_output[i].shape)
#     if eyebrow_morphing_combiner_onnx_output[i].shape[1] != 4:
#         continue
#     newIm = ((eyebrow_morphing_combiner_onnx_output[i].reshape(4,INPUT_SIZE*INPUT_SIZE).transpose().reshape(INPUT_SIZE,INPUT_SIZE,4) / 2.0 + 0.5)*255).astype('uint8')
#     im = Image.fromarray(newIm).convert('RGB')
#     im.show()

#Bench
from time import time 

in1 = im_morpher_crop.detach().numpy()
in2 = face_pose.detach().numpy()
t1 = time()
for i in range(10):
    face_morpher_onnx_res = ort_sess.run([            
            'output_image', #0
            'eye_alpha', #1
            'eye_color_change', #2
            'iris_mouth_image_1', #3
            'iris_mouth_alpha', #4
            'iris_mouth_color_change', #5
            'iris_mouth_image_0', #6
            ],
            {'eyebrow_image_no_combine_alpha': in1,
             "face_pose": in2
             })
t2 = time()
print("Run 10 times in onnx cost:", t2 - t1)

t1 = time()
for i in range(10):
    face_morpher(im_morpher_crop, face_pose)
t2 = time()
print("Run 10 times in pytorch cost:", t2 - t1)