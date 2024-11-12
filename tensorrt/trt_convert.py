from ctypes import cdll, c_char_p
libcudart = cdll.LoadLibrary('cudart64_12.dll')
libcudart.cudaGetErrorString.restype = c_char_p
def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)
cudaSetDevice(0) #No need but keep this when multi gpu case

import argparse
from pathlib import Path
import sys
from os.path import join
import numpy as np
import tensorrt as trt
from typing import List, Tuple
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, precision:str):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # Parse model file
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loading ONNX file from path {onnx_file_path}...')
    with open(onnx_file_path, 'rb') as model:
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Beginning ONNX file parsing')
        parse_res = parser.parse(model.read())
        if not parse_res:
            for error in range(parser.num_errors):
                TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed parsing of ONNX file')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Input number: {network.num_inputs}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Output number: {network.num_outputs}')
    def GiB(val):
        return val * 1 << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(1)) # 1G
    
    if precision == 'fp32':
        pass
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
    else:
        raise ValueError('precision must be one of fp32 or fp16')
        # Build engine.
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building an engine from file {onnx_file_path}; this may take a while...')
    serialized_engine = builder.build_serialized_network(network, config)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed creating Engine')
    return serialized_engine

def save_engine(engine, path):
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Saving engine to file {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(engine)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed saving engine')
def load_engine(path):
    TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {path}')
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')
    return engine

#memory management
class HostDeviceMem(object):
    def __init__(self, host_mem:numpy.ndarray, device_mem: pycuda.driver.DeviceAllocation):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    def __del__(self):
        self.device.free()
    def dtoh(self, stream:pycuda._driver.Stream):
        cuda.memcpy_dtoh_async(self.host, self.device, stream) 
    def htod(self, stream:pycuda._driver.Stream):
        cuda.memcpy_htod_async(self.device, self.host, stream)

class Processor:
    def __init__(self, engine: trt.ICudaEngine, n_input:int):
        self.engine = engine
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating inference context')
        # create execution context
        self.context = engine.create_execution_context()
        
        # get input and output tensor names
        self.input_tensor_names = [engine.get_tensor_name(i) for i in range(n_input)]
        self.output_tensor_names = [engine.get_tensor_name(i) for i in range(n_input, self.engine.num_io_tensors)]
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input nodes: '+ str(self.input_tensor_names))
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output nodes: '+ str(self.output_tensor_names))

        #create memories and bindings
        self.inputs = []
        self.outputs = []
        for bindingName in engine:
            shape = [dim for dim in self.context.get_tensor_shape(bindingName)]
            dtype = trt.nptype(engine.get_tensor_dtype(bindingName))
            host_mem = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.context.set_tensor_address(bindingName, int(device_mem)) # Use this setup without binding for v3
            if bindingName in self.input_tensor_names:
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

        # create stream
        self.stream = cuda.Stream()
        # Create a CUDA events
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()

            
    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)

    def loadInputs(self, inputs: List[np.ndarray]):
        # set input shapes, the output shapes are inferred automatically
        for inp, inp_mem in zip(inputs, self.inputs):
            if inp.dtype != inp_mem.host.dtype or inp.shape != inp_mem.host.shape:
                print('Given:', inp.dtype, inp.shape)
                print('Expected:',inp_mem.host.dtype, inp_mem.host.shape)
                raise ValueError('Input shape or type does not match')
            np.copyto(inp_mem.host, inp)
        for inp_mem in self.inputs: inp_mem.htod(self.stream)
        # Synchronize the stream
        self.stream.synchronize()

    def kickoff(self):
        # Record the start event
        self.start_event.record(self.stream)
        # Run inference.
        self.context.execute_async_v3(self.stream.handle)
        # Record the end event
        self.end_event.record(self.stream)
        # Synchronize the stream
        self.stream.synchronize()

    def extractOutputs(copy:bool = True) -> List[np.ndarray]:
        for out_mem in self.outputs: out_mem.dtoh(self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        if copy:
            return [np.copy(out.host) for outp in self.outputs]
        else:
            return [out.host for outp in self.outputs]
        
        
    def inference(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        inference process:
        1. create execution context
        2. set input shapes
        3. allocate memory
        4. copy input data to device
        5. run inference on device
        6. copy output data to host and reshape
        """

        # set input shapes, the output shapes are inferred automatically
        for inp, inp_mem in zip(inputs, self.inputs):
            if inp.dtype != inp_mem.host.dtype or inp.shape != inp_mem.host.shape:
                print('Given:', inp.dtype, inp.shape)
                print('Expected:',inp_mem.host.dtype, inp_mem.host.shape)
                raise ValueError('Input shape or type does not match')
            np.copyto(inp_mem.host, inp)

        for inp_mem in self.inputs: inp_mem.htod(self.stream)
            
        # Record the start event
        self.start_event.record(self.stream)
        # Run inference.
        self.context.execute_async_v3(self.stream.handle)
        # Record the end event
        self.end_event.record(self.stream)

        for out_mem in self.outputs: out_mem.dtoh(self.stream)
            
        # Synchronize the stream
        self.stream.synchronize()
        
        return [np.copy(outp.host) for outp in self.outputs]
    
    from os.path import join


def convert(sourceDir, dstDir, componentName, dataType):
    engine_seri = build_engine(join(sourceDir, componentName+'.onnx'), dataType)
    save_engine(engine_seri, join(dstDir, componentName+'.rtr'))

# convert('../onnx_model/fp16/', './fp16/','combiner', 'fp16')
# convert('../onnx_model/fp16/', './fp16/','decomposer', 'fp16')
# convert('../onnx_model/fp16/', './fp16/','editor', 'fp16')
# convert('../onnx_model/fp16/', './fp16/','morpher', 'fp16')
# convert('../onnx_model/fp16/', './fp16/','rotator', 'fp16')

# convert('../onnx_model/fp32/', './fp32/','combiner', 'fp32')
# convert('../onnx_model/fp32/', './fp32/','decomposer', 'fp32')
# convert('../onnx_model/fp32/', './fp32/','editor', 'fp32')
# convert('../onnx_model/fp32/', './fp32/','morpher', 'fp32')
# convert('../onnx_model/fp32/', './fp32/','rotator', 'fp32')

import onnxruntime as ort
providers = [("CUDAExecutionProvider", {"device_id": 0, #torch.cuda.current_device(),
                                        "user_compute_stream": str(int(cuda.Stream().handle))})]
sess_options = ort.SessionOptions()
def testVerify(sourceDir, dstDir, componentName, dataType, inputNames, inputShapes):
    if dataType == 'fp16':
        dtype = np.float16  
    elif dataType == 'fp32':
        dtype = np.float32
    else:
        dtype = np.int8
    inputs = [np.random.rand(*shape).astype(dtype) for shape in inputShapes]
    input_dict = {k:v for k,v in zip(inputNames,inputs)}
    
    engine = load_engine(join(dstDir, componentName+'.rtr'))
    proc = Processor(engine, len(inputNames))
    trt_res = proc.inference(inputs)

    ort_sess = ort.InferenceSession(join(sourceDir, componentName+'.onnx'), sess_options=sess_options, providers=providers)
    onnx_res = ort_sess.run(None,input_dict)
    for i in range(len(trt_res)):
        print("MSE is: ",((trt_res[i] - onnx_res[i]) ** 2).mean())


print("Decomposer:")
decomposer_input_names = ['input_image']
decomposer_input_shapes = [(1,4,512,512)]
# testVerify('../onnx_model/fp32/', './fp32/','decomposer', 'fp32', decomposer_input_names, decomposer_input_shapes)
# testVerify('../onnx_model/fp16/', './fp16/','decomposer', 'fp16', decomposer_input_names, decomposer_input_shapes)

print("Combiner:")
combiner_input_names = ['input_image','eyebrow_background_layer', "eyebrow_layer", 'eyebrow_pose']
combiner_input_shapes = [(1,4,512,512), (1,4,128,128), (1,4,128,128), (1,12)]
# testVerify('../onnx_model/fp32/', './fp32/','combiner', 'fp32', combiner_input_names, combiner_input_shapes)
# testVerify('../onnx_model/fp16/', './fp16/','combiner', 'fp16', combiner_input_names, combiner_input_shapes)

print("Morpher:")
morpher_input_names = ['input_image', 'im_morpher_crop', 'face_pose', '/face_morpher/body/downsample_blocks.3/downsample_blocks.3.3/Relu_output_0']
morpher_input_shapes = [(1,4,512,512), (1,4,192,192), (1,27), (1,512,24,24)]
# testVerify('../onnx_model/fp32/', './fp32/','morpher', 'fp32', morpher_input_names, morpher_input_shapes)
# testVerify('../onnx_model/fp16/', './fp16/','morpher', 'fp16', morpher_input_names, morpher_input_shapes)

print("Rotator:")
rotator_input_shapes = [(1,4,256,256), (1,6)]
rotator_input_names = ['face_morphed_half', 'rotation_pose']
# testVerify('../onnx_model/fp32/', './fp32/','rotator', 'fp32', rotator_input_names, rotator_input_shapes)
# testVerify('../onnx_model/fp16/', './fp16/','rotator', 'fp16', rotator_input_names, rotator_input_shapes)

print("Editor:")
editor_input_shapes = [(1,4,512,512), (1,4,512,512), (1,2,512,512), (1,6)]
editor_input_names = ['morphed_image', 'rotated_warped_image','rotated_grid_change','rotation_pose']
# testVerify('../onnx_model/fp32/', './fp32/','editor', 'fp32', editor_input_names, editor_input_shapes)
# testVerify('../onnx_model/fp16/', './fp16/','editor', 'fp16', editor_input_names, editor_input_shapes)


model_dir = './fp16/'
dtype = np.float16

decomposer_engine = load_engine(join(model_dir, 'decomposer.rtr'))
decomposer_proc = Processor(decomposer_engine, 4)

combiner_engine = load_engine(join(model_dir, 'combiner.rtr'))
combiner_proc = Processor(combiner_engine, 4)

morpher_engine = load_engine(join(model_dir, 'morpher.rtr'))
morpher_proc = Processor(morpher_engine, 4)
morpher_proc.loadInputs([np.random.rand(*shape).astype(dtype) for shape in morpher_input_shapes])

rotator_engine = load_engine(join(model_dir, 'rotator.rtr'))
rotator_proc = Processor(rotator_engine, 2)
rotator_proc.loadInputs([np.random.rand(*shape).astype(dtype) for shape in rotator_input_shapes])

tmp = [np.random.rand(*shape).astype(dtype) for shape in rotator_input_shapes]

editor_engine = load_engine(join(model_dir, 'editor.rtr'))
editor_proc = Processor(editor_engine, 2)
editor_proc.loadInputs([np.random.rand(*shape).astype(dtype) for shape in editor_input_shapes])

rife_engine = load_engine('../RIFE/rife.trt')
rife_proc = Processor(rife_engine, 2)


for i in range(10): #Preheat
    morpher_proc.kickoff()
    rotator_proc.loadInputs(tmp)
    rotator_proc.kickoff()
    editor_proc.kickoff()

from time import time,sleep
from tqdm import tqdm
for i in tqdm(range(100000000)):
    t1 = time()
    morpher_proc.kickoff()
    rotator_proc.loadInputs(tmp)
    rotator_proc.kickoff()
    editor_proc.kickoff()

    rife_proc.kickoff()
    t2 = 0.060 - (time()-t1)
    if t2 >0.0:
        sleep(t2)