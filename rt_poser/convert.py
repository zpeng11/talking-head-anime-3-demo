import sys
import os
sys.path.append(os.getcwd())
from trt.trt_utils import *

def convert(dest_dir, model_type, component, dtype):
    src_dir = join('.','onnx_model', model_type, dtype , component + '.onnx')
    os.makedirs(dest_dir, exist_ok=True)
    des = join(dest_dir, component + '.trt')
    engine_seri = build_engine(src_dir, dtype)
    save_engine(engine_seri, des)

if __name__ == "__main__":
    # convert('./rt_poser/seperable/fp16','seperable', 'combiner', 'fp16')
    # convert('./rt_poser/seperable/fp16','seperable', 'decomposer', 'fp16')
    # convert('./rt_poser/seperable/fp16','seperable', 'editor', 'fp16')
    # convert('./rt_poser/seperable/fp16','seperable', 'morpher', 'fp16')
    # convert('./rt_poser/seperable/fp16','seperable', 'rotator', 'fp16')

    # convert('./rt_poser/seperable/fp32','seperable', 'combiner', 'fp32')
    # convert('./rt_poser/seperable/fp32','seperable', 'decomposer', 'fp32')
    # convert('./rt_poser/seperable/fp32','seperable', 'editor', 'fp32')
    # convert('./rt_poser/seperable/fp32','seperable', 'morpher', 'fp32')
    # convert('./rt_poser/seperable/fp32','seperable', 'rotator', 'fp32')

    convert('./rt_poser/standard/fp32','standard' , 'combiner', 'fp32')
    convert('./rt_poser/standard/fp32','standard' , 'decomposer', 'fp32')
    convert('./rt_poser/standard/fp32','standard' , 'editor', 'fp32')
    convert('./rt_poser/standard/fp32','standard' , 'morpher', 'fp32')
    convert('./rt_poser/standard/fp32','standard' , 'rotator', 'fp32')

    convert('./rt_poser/standard/fp16','standard' , 'combiner', 'fp16')
    convert('./rt_poser/standard/fp16','standard' , 'decomposer', 'fp16')
    convert('./rt_poser/standard/fp16','standard' , 'editor', 'fp16')
    convert('./rt_poser/standard/fp16','standard' , 'morpher', 'fp16')
    convert('./rt_poser/standard/fp16','standard' , 'rotator', 'fp16')

