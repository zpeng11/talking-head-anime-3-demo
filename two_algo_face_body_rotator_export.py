import torch
import numpy
import PIL.Image
from tha3.util import resize_PIL_image,extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image
from torch.nn.functional import interpolate

MODEL_NAME = "standard_float"
DEVICE_NAME = 'cpu'
IMAGE_INPUT = "data\images\crypko_03.png"
ONNX_MODEL_NAME = "two_algo_face_body_rotator.onnx"


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
two_algo_face_body_rotator = poser.get_modules()['two_algo_face_body_rotator']

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

face_morphed_full = im.clone()
face_morphed_full[:, :, 32:32 + 192, 32 + 128:32 + 192 + 128] = face_morpher_torch_res[0]
face_morphed_half = interpolate(face_morphed_full, size=(256, 256), mode='bilinear', align_corners=False)
rotation_pose = torch.zeros((1,6))

two_algo_face_body_rotator_torch_res = two_algo_face_body_rotator(face_morphed_half, rotation_pose)

torch.onnx.export(two_algo_face_body_rotator,               # model being run
                  (face_morphed_half, rotation_pose),                         # model input (or a tuple for multiple inputs)
                  ONNX_MODEL_NAME,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['face_morphed_half', 'rotation_pose'],   # the model's input names
                  output_names = [  
            'direct_image',
            'warped_image',
            'grid_change'
            ]) 

import onnx
onnx_model = onnx.load(ONNX_MODEL_NAME)
onnx.checker.check_model(onnx_model)

import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession(ONNX_MODEL_NAME)
two_algo_face_body_rotator_onnx_res = ort_sess.run([            
            'direct_image',
            'warped_image',
            'grid_change'
            ],
            {'face_morphed_half': face_morphed_half.detach().numpy(),
             "rotation_pose":rotation_pose.detach().numpy()
             })


# from PIL import Image
# INPUT_SIZE = 128
for i in range(len(two_algo_face_body_rotator_onnx_res)):
    print("MSE is: ",((two_algo_face_body_rotator_onnx_res[i] - two_algo_face_body_rotator_torch_res[i].detach().numpy()) ** 2).mean())
#     print(eyebrow_morphing_combiner_onnx_output[i].shape)
#     if eyebrow_morphing_combiner_onnx_output[i].shape[1] != 4:
#         continue
#     newIm = ((eyebrow_morphing_combiner_onnx_output[i].reshape(4,INPUT_SIZE*INPUT_SIZE).transpose().reshape(INPUT_SIZE,INPUT_SIZE,4) / 2.0 + 0.5)*255).astype('uint8')
#     im = Image.fromarray(newIm).convert('RGB')
#     im.show()