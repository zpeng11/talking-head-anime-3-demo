import torch
import os
from os.path import join
import sys
from pathlib import Path
sys.path.insert(0, join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy
import numpy as np
import PIL.Image
from tha3.util import resize_PIL_image,extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image
from tqdm import tqdm
import onnx
from onnxsim import simplify
import onnxruntime as ort
import numpy as np
from torch import Tensor
from torch.nn import Module
from typing import List, Optional
from torch.nn.functional import interpolate
import onnx_tool



MODEL_NAME = "separable_float"
HALF = False
DEVICE_NAME = 'cuda:0'
IMAGE_INPUT = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','data','images','crypko_07.png')
USE_RANDOM_IMAGE = False
TMP_DIR = join(os.path.dirname(os.path.realpath(__file__)),'tmp')
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
MODEL_DIR = join(os.path.dirname(os.path.realpath(__file__)),'fp32')
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
TMP_FILE_WRITE = join(TMP_DIR, 'tmp.onnx')

providers = [("CUDAExecutionProvider", {"device_id": 0, #torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
sess_options = ort.SessionOptions()
sess_options.log_severity_level=1

device = torch.device(DEVICE_NAME)
dtype = torch.float16 if HALF else torch.float32

#Prepare models
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
editor = poser.get_modules()['editor']

#Prepare one pass inference image data
pt_img = None
if USE_RANDOM_IMAGE:
    pt_img = torch.rand(1, 4, 512, 512,dtype=dtype, device=device) * 2.0 - 1.0
else:
    pil_image = resize_PIL_image(extract_PIL_image_from_filelike(IMAGE_INPUT), size=(512,512))
    
    if HALF:
        pt_img = extract_pytorch_image_from_PIL_image(pil_image).half().reshape(1,4,512,512).to(DEVICE_NAME)
    else:
        pt_img = extract_pytorch_image_from_PIL_image(pil_image).reshape(1,4,512,512).to(DEVICE_NAME)
zero_pose = torch.zeros(1, pose_size, dtype=dtype, device=device)

poser_torch_res = poser.pose(pt_img, zero_pose)




class EyebrowDecomposerWrapper(Module):
    def __init__(self, eyebrow_decomposer_obj):
        super().__init__()
        self.eyebrow_decomposer = eyebrow_decomposer_obj
    def forward(self, image: Tensor, *args) -> List[Tensor]:
        cropped = image[:,:, 64:192, 64 + 128:192 + 128].reshape((1,4,128,128))
        decomposer_res = self.eyebrow_decomposer(cropped)
        return [decomposer_res[3], decomposer_res[0]]
eyebrow_decomposer_wrapper = EyebrowDecomposerWrapper(eyebrow_decomposer).eval()
eyebrow_decomposer_wrapped_torch_res = eyebrow_decomposer_wrapper(pt_img)

EYEBROW_DECOMPOSER_INPUT_LIST = ['input_image']
EYEBROW_DECOMPOSER_OUTPUT_LIST = ["background_layer", "eyebrow_layer"]
#Export onnx model finally get a simplified decomposer onnx model
torch.onnx.export(eyebrow_decomposer_wrapper,               # model being run
                  pt_img,                         # model input (or a tuple for multiple inputs)
                  TMP_FILE_WRITE,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = EYEBROW_DECOMPOSER_INPUT_LIST,   # the model's input names
                  output_names = EYEBROW_DECOMPOSER_OUTPUT_LIST) # the model's output names
onnx_model = onnx.load(TMP_FILE_WRITE)
onnx.checker.check_model(onnx_model)
onnx_model_sim, check = simplify(onnx_model)
if check:
    onnx.save(onnx_model_sim, join(MODEL_DIR, "decomposer.onnx"))
else:
    print("Simplify error!")



EYEBROW_MORPHING_COMBINER_INPUT_LIST = ['input_image','eyebrow_background_layer', "eyebrow_layer", 'eyebrow_pose']
EYEBROW_MORPHING_COMBINER_OUTPUT_LIST = ['eyebrow_image']  
EYEBROW_POSE_SHAPE = (1, 12)
eyebrow_pose_zero = torch.zeros(EYEBROW_POSE_SHAPE, dtype=dtype, device=device)

#Build a new eyebrow_morphing_combiner that does cropping
class EyebrowMorphingCombinerWrapper(Module):
    def __init__(self, eyebrow_morphing_combiner_obj):
        super().__init__()
        self.eyebrow_morphing_combiner = eyebrow_morphing_combiner_obj
    def forward(self, full_image:Tensor, background_layer: Tensor, eyebrow_layer: Tensor, pose: Tensor, *args) -> Tensor:
        im_morpher_crop = full_image[:, :, 32:32 + 192, (32 + 128):(32 + 192 + 128)].clone()
        im_morpher_crop[:, :, 32:32 + 128, 32:32 + 128] = self.eyebrow_morphing_combiner(background_layer, eyebrow_layer, pose)[2]
        return im_morpher_crop
eyebrow_morphing_combiner_wrapped = EyebrowMorphingCombinerWrapper(eyebrow_morphing_combiner).eval()
eyebrow_morphing_combiner_wrapped_torch_res = eyebrow_morphing_combiner_wrapped(pt_img, eyebrow_decomposer_wrapped_torch_res[0], 
                                                                                eyebrow_decomposer_wrapped_torch_res[1], eyebrow_pose_zero)
input_tuple = (pt_img, eyebrow_decomposer_wrapped_torch_res[0], eyebrow_decomposer_wrapped_torch_res[1], eyebrow_pose_zero)
torch.onnx.export(eyebrow_morphing_combiner_wrapped,               # model being run
                  input_tuple,                         # model input (or a tuple for multiple inputs)
                  TMP_FILE_WRITE,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = EYEBROW_MORPHING_COMBINER_INPUT_LIST,   # the model's input names
                  output_names = EYEBROW_MORPHING_COMBINER_OUTPUT_LIST) 
onnx_model = onnx.load(TMP_FILE_WRITE)
onnx.checker.check_model(onnx_model)
onnx_model_sim, check = simplify(onnx_model)
if check:
    onnx.save(onnx_model_sim,join(TMP_DIR, 'combiner_tmp.onnx')) #Will update later
else:
    print("Simplify error!")


#Play with face morpher
FACE_POSE_SHAPE = (1,27)
face_pose_zero = torch.zeros(FACE_POSE_SHAPE, dtype=dtype, device=device)

class FaceMorpherWrapped(Module):
    def __init__(self, face_morpher_obj):
        super().__init__()
        self.face_morpher = face_morpher_obj
    def forward(self, input_image: Tensor, im_morpher_crop: Tensor, face_pose:Tensor,  *args) -> List[Tensor]:
        face_morphed_full = input_image.clone()
        face_morphed_full[:, :, 32:32 + 192, 32 + 128:32 + 192 + 128] = self.face_morpher(im_morpher_crop, face_pose)[0]
        face_morphed_half = interpolate(face_morphed_full, size=(256, 256), mode='bilinear', align_corners=False)
        return [face_morphed_full, face_morphed_half]
face_morpher_wrapped = FaceMorpherWrapped(face_morpher).eval()
face_morpher_wrapped_torch_res = face_morpher_wrapped(pt_img, eyebrow_morphing_combiner_wrapped_torch_res, face_pose_zero) 

FACE_MORPHER_OUTPUT_LIST = ['face_morphed_full', 'face_morphed_half']
FACE_MORPHER_INPUT_LIST = ['input_image', 'im_morpher_crop', 'face_pose']
input_tuple = (pt_img, eyebrow_morphing_combiner_wrapped_torch_res, face_pose_zero)

torch.onnx.export(face_morpher_wrapped,               # model being run
                  input_tuple,                         # model input (or a tuple for multiple inputs)
                  TMP_FILE_WRITE,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = FACE_MORPHER_INPUT_LIST,   # the model's input names
                  output_names = FACE_MORPHER_OUTPUT_LIST) 

onnx_model = onnx.load(TMP_FILE_WRITE)
onnx.checker.check_model(onnx_model)
onnx_model_sim, check = simplify(onnx_model)
if check:
    onnx.save(onnx_model_sim,join(TMP_DIR, 'morpher_tmp.onnx'))
else:
    print("Simplify error!")

# Try to split out the Encoder part of the mopher model
FACE_MORPHER_ENCODER = join(TMP_DIR, 'morpher_encoder.onnx')
onnx.utils.extract_model(join(TMP_DIR, 'morpher_tmp.onnx'), FACE_MORPHER_ENCODER, ['im_morpher_crop'], 
                         ['/face_morpher/body/downsample_blocks.3/downsample_blocks.3.3/Relu_output_0'])
onnx.checker.check_model(onnx.load(FACE_MORPHER_ENCODER))
FACE_MORPHER_NEW = join(MODEL_DIR, 'morpher.onnx')
onnx.utils.extract_model(join(TMP_DIR, 'morpher_tmp.onnx'), FACE_MORPHER_NEW, 
                         ['input_image','im_morpher_crop','face_pose',
                          '/face_morpher/body/downsample_blocks.3/downsample_blocks.3.3/Relu_output_0'], 
                         ['face_morphed_full', 'face_morphed_half'])
onnx.checker.check_model(onnx.load(FACE_MORPHER_NEW))

EYEBROW_COMBINER_NEW = join(MODEL_DIR, 'combiner.onnx')
eyebrow_combiner_model =  onnx.load(join(TMP_DIR, 'combiner_tmp.onnx'))
face_morpher_encoder_model = onnx.load(FACE_MORPHER_ENCODER)

eyebrow_combiner_new_model = onnx.compose.merge_models(
    eyebrow_combiner_model, face_morpher_encoder_model,
    io_map=[("eyebrow_image", "im_morpher_crop")]
)
onnx.save(eyebrow_combiner_new_model, TMP_FILE_WRITE)
onnx.utils.extract_model(TMP_FILE_WRITE, EYEBROW_COMBINER_NEW, ['input_image', 'eyebrow_background_layer', 'eyebrow_layer', 'eyebrow_pose'], 
                         ['eyebrow_image', '/face_morpher/body/downsample_blocks.3/downsample_blocks.3.3/Relu_output_0'])
onnx.checker.check_model(onnx.load(EYEBROW_COMBINER_NEW))

ROTATION_POSE_SHAPE = (1,6)
rotation_pose_zero = torch.zeros(ROTATION_POSE_SHAPE, dtype=dtype, device=device)

class TwoAlgoFaceBodyRotatorWrapped(Module):
    def __init__(self, two_algo_face_body_rotator_obj):
        super().__init__()
        self.two_algo_face_body_rotator = two_algo_face_body_rotator_obj
    def forward(self, image: Tensor, pose: Tensor, *args) -> List[Tensor]:
        res = self.two_algo_face_body_rotator(image, pose)
        full_warped_image = interpolate(res[1], size=(512, 512), mode='bilinear', align_corners=False)
        full_grid_change = interpolate(res[2], size=(512, 512), mode='bilinear', align_corners=False)
        return [full_warped_image, full_grid_change]
two_algo_face_body_rotator_wrapped = TwoAlgoFaceBodyRotatorWrapped(two_algo_face_body_rotator).eval()
rotator_wrapped_torch_res = two_algo_face_body_rotator_wrapped(face_morpher_wrapped_torch_res[1], rotation_pose_zero)

ROTATOR_OUTPUT_LIST = ['full_warped_image', 'full_grid_change']
ROTATOR_INPUT_LIST = ['face_morphed_half', 'rotation_pose']
input_tuple = (face_morpher_wrapped_torch_res[1], rotation_pose_zero)

torch.onnx.export(two_algo_face_body_rotator_wrapped,               # model being run
                  input_tuple,                         # model input (or a tuple for multiple inputs)
                  TMP_FILE_WRITE,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ROTATOR_INPUT_LIST,   # the model's input names
                  output_names = ROTATOR_OUTPUT_LIST) 

onnx_model = onnx.load(TMP_FILE_WRITE)
onnx.checker.check_model(onnx_model)
onnx_model_sim, check = simplify(onnx_model)
if check:
    onnx.save(onnx_model_sim,join(MODEL_DIR, 'rotator.onnx'))
else:
    print("Simplify error!")

class EditorWrapped(Module):
    def __init__(self, editor_obj):
        super().__init__()
        self.editor = editor_obj
    def forward(self,
                morphed_image: Tensor,
                rotated_warped_image: Tensor,
                rotated_grid_change: Tensor,
                pose: Tensor,
                *args) -> List[Tensor]:
        return self.editor(morphed_image, rotated_warped_image, rotated_grid_change, pose)[0]
editor_wrapped = EditorWrapped(editor).eval()
editor_wrapped_torch_res = editor_wrapped(face_morpher_wrapped_torch_res[0], 
                                          rotator_wrapped_torch_res[0], 
                                          rotator_wrapped_torch_res[1], 
                                          rotation_pose_zero)

EDITOR_OUTPUT_LIST = ['result']
EDITOR_INPUT_LIST = ['morphed_image', 'rotated_warped_image','rotated_grid_change','rotation_pose']
input_tuple = (face_morpher_wrapped_torch_res[0], 
                          rotator_wrapped_torch_res[0], 
                          rotator_wrapped_torch_res[1], 
                          rotation_pose_zero)

torch.onnx.export(editor_wrapped,               # model being run
                  input_tuple,                         # model input (or a tuple for multiple inputs)
                  TMP_FILE_WRITE,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = EDITOR_INPUT_LIST,   # the model's input names
                  output_names = EDITOR_OUTPUT_LIST) 

onnx_model = onnx.load(TMP_FILE_WRITE)
onnx.checker.check_model(onnx_model)
onnx_model_sim, check = simplify(onnx_model)
if check:
    onnx.save(onnx_model_sim,join(MODEL_DIR, 'editor.onnx'))
else:
    print("Simplify error!")


class RunTest():
    def __init__(self, img = None, ref = None):
        if HALF:
            self.dtype = np.float16
        else:
            self.dtype = np.float32
        self.decomposer_sess = ort.InferenceSession(join(MODEL_DIR, 'decomposer.onnx'), sess_options=sess_options, providers=providers)
        self.combiner_sess = ort.InferenceSession(join(MODEL_DIR, "combiner.onnx"), sess_options=sess_options, providers=providers)
        self.morpher_sess = ort.InferenceSession(join(MODEL_DIR, "morpher.onnx"), sess_options=sess_options, providers=providers)
        self.rotator_sess = ort.InferenceSession(join(MODEL_DIR, "rotator.onnx"), sess_options=sess_options, providers=providers)
        self.editor_sess = ort.InferenceSession(join(MODEL_DIR, "editor.onnx"), sess_options=sess_options, providers=providers)
        if img == None:
            img = np.random.rand(1, 4, 512, 512).astype(self.dtype) * 2.0 - 1.0
        else:
            img = img.cpu().detach().numpy()
        self.eyebrow_pose_zero = np.zeros((1,12), dtype=self.dtype)
        self.face_pose_zero = np.zeros((1,27), dtype=self.dtype)
        self.rotation_pose_zero = np.zeros((1,6), dtype=self.dtype)

        decomposer_res = self.decomposer_sess.run(None, {'input_image':img,})
        combiner_res = self.combiner_sess.run(None, {'input_image':img,
                                                     'eyebrow_background_layer': decomposer_res[0],
                                                     "eyebrow_layer": decomposer_res[1],
                                                     'eyebrow_pose':self.eyebrow_pose_zero,})
        morpher_res = self.morpher_sess.run(None, {'input_image':img,
                                                   'im_morpher_crop': combiner_res[0], 
                                                   'face_pose': self.face_pose_zero,
                                                   '/face_morpher/body/downsample_blocks.3/downsample_blocks.3.3/Relu_output_0':combiner_res[1]})
        rotator_res = self.rotator_sess.run(None, {'face_morphed_half':morpher_res[1], 
                                                   'rotation_pose':self.rotation_pose_zero})
        editor_res = self.editor_sess.run(None, {'morphed_image':morpher_res[0],
                                                 'rotated_warped_image':rotator_res[0],
                                                 'rotated_grid_change': rotator_res[1], 
                                                 'rotation_pose':self.rotation_pose_zero})
        if ref != None:
            def printInfo(a):
                print(a.dtype, a.shape, np.max(a),np.min(a), np.mean(a), np.sum(a))
            ref_np = ref.cpu().detach().numpy()
            printInfo(editor_res[0])
            printInfo(ref_np)
            print("MSE is: ",((editor_res[0] - ref_np) ** 2).mean())
            # from PIL import Image
            # def saveImg(path:str, arry):
            #     resImg = ((arry/2.0 + 0.5)*255).astype('uint8')
            #     Image.fromarray(resImg).convert('RGB').save(path)
            # saveImg('test_res.jpg',editor_res[0])
            # saveImg('ref.jpg', ref_np)
            
RunTest(pt_img, poser_torch_res[0])