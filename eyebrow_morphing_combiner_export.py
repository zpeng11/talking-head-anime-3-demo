import torch
import numpy
import PIL.Image
from tha3.util import resize_PIL_image,extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image

MODEL_NAME = "standard_float"
DEVICE_NAME = 'cpu'
IMAGE_INPUT = "data\images\crypko_03.png"
ONNX_MODEL_NAME = "eyebrow_morphing_combiner.onnx"


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

def load_input_img():
    pil_image = resize_PIL_image(extract_PIL_image_from_filelike(IMAGE_INPUT), size=(512,512))
    return extract_pytorch_image_from_PIL_image(pil_image).to(device)

im = load_input_img().reshape(1,4,512,512)
# im = torch.rand(1, 4, 512, 512) * 2.0 - 1.0
pose = torch.zeros(1, pose_size, dtype=poser.get_dtype())
poser.pose(im, pose)

im = im[:,:, 64:192, 64 + 128:192 + 128]

eyebrow_pose = torch.zeros((1, 12))

eyebrow_decomposer_torch_res = eyebrow_decomposer(im)
print(eyebrow_decomposer_torch_res[3].shape, eyebrow_decomposer_torch_res[0].shape)

eyebrow_morphing_combiner_torch_res = eyebrow_morphing_combiner(eyebrow_decomposer_torch_res[3], eyebrow_decomposer_torch_res[0], eyebrow_pose)


torch.onnx.export(eyebrow_morphing_combiner,               # model being run
                  (eyebrow_decomposer_torch_res[3], eyebrow_decomposer_torch_res[0], eyebrow_pose),                         # model input (or a tuple for multiple inputs)
                  ONNX_MODEL_NAME,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['eyebrow_background_layer', "eyebrow_layer", 'eyebrow_pose'],   # the model's input names
                  output_names = [  
            'eyebrow_image',  # 0
            'combine_alpha',  # 1
            'eyebrow_image_no_combine_alpha',  # 2
            'morphed_eyebrow_layer',  # 3
            'morphed_eyebrow_layer_alpha',  # 4
            'morphed_eyebrow_layer_color_change',  # 5
            'warped_eyebrow_layer',  # 6
            'morphed_eyebrow_layer_grid_change',  # 7
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
eyebrow_morphing_combiner_onnx_output = ort_sess.run([            
            'eyebrow_image',  # 0
            'combine_alpha',  # 1
            'eyebrow_image_no_combine_alpha',  # 2
            'morphed_eyebrow_layer',  # 3
            'morphed_eyebrow_layer_alpha',  # 4
            'morphed_eyebrow_layer_color_change',  # 5
            'warped_eyebrow_layer',  # 6
            'morphed_eyebrow_layer_grid_change',  # 7
            ],
            {'eyebrow_background_layer': eyebrow_decomposer_torch_res[3].detach().numpy(),
             "eyebrow_layer":eyebrow_decomposer_torch_res[0].detach().numpy(),
             'eyebrow_pose':eyebrow_pose.detach().numpy()})


from PIL import Image
INPUT_SIZE = 128
for i in range(len(eyebrow_morphing_combiner_onnx_output)):
    print("MSE is: ",((eyebrow_morphing_combiner_onnx_output[i] - eyebrow_morphing_combiner_torch_res[i].detach().numpy()) ** 2).mean())
    print(eyebrow_morphing_combiner_onnx_output[i].shape)
    if eyebrow_morphing_combiner_onnx_output[i].shape[1] != 4:
        continue
    newIm = ((eyebrow_morphing_combiner_onnx_output[i].reshape(4,INPUT_SIZE*INPUT_SIZE).transpose().reshape(INPUT_SIZE,INPUT_SIZE,4) / 2.0 + 0.5)*255).astype('uint8')
    im = Image.fromarray(newIm).convert('RGB')
    im.show()