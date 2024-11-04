import torch
import numpy
import PIL.Image
from tha3.util import resize_PIL_image,extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image

MODEL_NAME = "standard_float"
DEVICE_NAME = 'cpu'
IMAGE_INPUT = "data\images\crypko_03.png"
ONNX_MODEL_NAME = "eyebrow_decomposer.onnx"


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

def load_input_img():
    pil_image = resize_PIL_image(extract_PIL_image_from_filelike(IMAGE_INPUT), size=(512,512))
    return extract_pytorch_image_from_PIL_image(pil_image).to(device)

im = load_input_img()

im = im[:, 64:192, 64 + 128:192 + 128].reshape(1,4,128,128)
INPUT_SIZE = 128

torch_res = eyebrow_decomposer(im)
for res in torch_res:
    print(res.shape, res.max(), res.min())

print(im.dtype)
print(im.shape)

for m in eyebrow_decomposer.modules(): 
    if 'instancenorm' in m.__class__.__name__.lower(): m.train(False)

torch.onnx.export(eyebrow_decomposer,               # model being run
                  im,                         # model input (or a tuple for multiple inputs)
                  ONNX_MODEL_NAME,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = [            "eyebrow_layer",  # 0
            "eyebrow_layer_alpha",  # 1
            "eyebrow_layer_color_change",  # 2
            "background_layer_1",  # 3
            "background_layer_alpha",  # 4
            "background_layer_color_change"  # 5
            ]) # the model's output names

import onnx
onnx_model = onnx.load(ONNX_MODEL_NAME)
onnx.checker.check_model(onnx_model)
from onnxsim import simplify

assert check,"Simply is not avaialable"
onnx.save(onnx_model_sim,"sim_"+ONNX_MODEL_NAME)

import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession("sim_"+ONNX_MODEL_NAME)
onnx_output = ort_sess.run([            "eyebrow_layer",  # 0
            "eyebrow_layer_alpha",  # 1
            "eyebrow_layer_color_change",  # 2
            "background_layer_1",  # 3
            "background_layer_alpha",  # 4
            "background_layer_color_change"  # 5
            ],{'input': im.numpy()})

from PIL import Image

for i in range(len(onnx_output)):
    print("MSE is: ",((onnx_output[i] - torch_res[i].detach().numpy()) ** 2).mean())
    if onnx_output[i].shape[0] == 1:
        continue
    newIm = ((onnx_output[i].reshape(4,INPUT_SIZE*INPUT_SIZE).transpose().reshape(INPUT_SIZE,INPUT_SIZE,4) / 2.0 + 0.5)*255).astype('uint8')
    print(newIm.shape, type(newIm), newIm.min(), newIm.max())
    im = Image.fromarray(newIm).convert('RGB')
    im.show()


# onnx_program = torch.onnx.export(model, torch_input)