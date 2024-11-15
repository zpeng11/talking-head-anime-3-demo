import sys
import os
sys.path.append(os.getcwd())
from trt.trt_utils import *
from os.path import join
import numpy as np
from abc import ABC

class NodeInfo:
    def __init__(self, name:str, shape:List[int], dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
    def __str__(self) -> str:
        return self.name+": "+str(self.shape) + " "+str(self.dtype)
    def __repr__(self):
        return self.__str__()
    
def createMemory(nodeInfo : NodeInfo):
    shape = nodeInfo.shape
    dtype = nodeInfo.dtype
    host_mem = cuda.pagelocked_empty(shape, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    return HostDeviceMem(host_mem, device_mem)

class Engine:
    def __init__(self, path:str, n_inputs:int):
        TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {path}')
        runtime = trt.Runtime(TRT_LOGGER)
        with open(path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')

        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating inference context')
        self.context = self.engine.create_execution_context()
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed inference context')
        
        self.inputs = []
        for i in range(n_inputs):
            name = self.engine.get_tensor_name(i)
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output node: '+ name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.inputs.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.inputs))

        self.outputs = []
        for i in range(n_inputs, self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output node: '+ name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.outputs.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.outputs))

    def setInputMems(self, inputMems:List[HostDeviceMem]): # inputs pass by reference
        assert(len(inputMems) == len(self.inputs))
        for i in range(len(inputMems)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.inputs[i].name)
            self.context.set_tensor_address(self.inputs[i].name, int(inputMems[i].device)) # Use this setup without binding for v3

    def setOutputMems(self, outputMems:List[HostDeviceMem]): # outputs pass by reference
        assert(len(outputMems) == len(self.outputs))
        for i in range(len(outputMems)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.outputs[i].name)
            self.context.set_tensor_address(self.outputs[i].name, int(outputMems[i].device)) # Use this setup without binding for v3
    def exec(self, stream):
        self.context.execute_async_v3(stream.handle)

class RTCoreBase(ABC):
    def __init__(self, model_dir):
        self.prepareProcs(model_dir)
        self.prepareMemories()
        self.setMemsToEngines()

    def inference(self):
        return

    def prepareProcs(self, model_dir):
        self.decomposer = Engine(join(model_dir, 'decomposer.trt'), 1)
        self.combiner = Engine(join(model_dir, 'combiner.trt'), 4)
        self.morpher = Engine(join(model_dir, 'morpher.trt'), 4)
        self.rotator = Engine(join(model_dir, 'rotator.trt'), 2)
        self.editor = Engine(join(model_dir, 'editor.trt'), 4)

    def prepareMemories(self):
        self.memories = {}
        self.memories['input_img'] = createMemory(self.decomposer.inputs[0])
        self.memories["background_layer"] = createMemory(self.decomposer.outputs[0])
        self.memories["eyebrow_layer"] = createMemory(self.decomposer.outputs[1])

        self.memories['eyebrow_pose'] = createMemory(self.combiner.inputs[3])
        self.memories['eyebrow_image'] = createMemory(self.combiner.outputs[0])
        self.memories['morpher_decoded'] = createMemory(self.combiner.outputs[1])

        self.memories['face_pose'] = createMemory(self.morpher.inputs[2])
        self.memories['face_morphed_full'] = createMemory(self.morpher.outputs[0])
        self.memories['face_morphed_half'] = createMemory(self.morpher.outputs[1])

        self.memories['rotation_pose'] = createMemory(self.rotator.inputs[1])
        self.memories['wrapped_image'] = createMemory(self.rotator.outputs[0])
        self.memories['grid_change'] = createMemory(self.rotator.outputs[1])

        self.memories['output_img'] = createMemory(self.editor.outputs[0])

    def setMemsToEngines(self):
        decomposer_inputs = [self.memories['input_img']]
        self.decomposer.setInputMems(decomposer_inputs)
        decomposer_outputs = [self.memories["background_layer"], self.memories["eyebrow_layer"]]
        self.decomposer.setOutputMems(decomposer_outputs)

        combiner_inputs = [self.memories['input_img'], self.memories["background_layer"], self.memories["eyebrow_layer"], self.memories['eyebrow_pose']]
        self.combiner.setInputMems(combiner_inputs)
        combiner_outputs = [self.memories['eyebrow_image'], self.memories['morpher_decoded']]
        self.combiner.setOutputMems(combiner_outputs)

        morpher_inputs = [self.memories['input_img'], self.memories['eyebrow_image'], self.memories['face_pose'], self.memories['morpher_decoded']]
        self.morpher.setInputMems(morpher_inputs)
        morpher_outputs = [self.memories['face_morphed_full'], self.memories['face_morphed_half']]
        self.morpher.setOutputMems(morpher_outputs)

        rotator_inputs = [self.memories['face_morphed_half'], self.memories['rotation_pose']]
        self.rotator.setInputMems(rotator_inputs)
        rotator_outputs = [self.memories['wrapped_image'], self.memories['grid_change']]
        self.rotator.setOutputMems(rotator_outputs)

        editor_inputs = [self.memories['face_morphed_full'], self.memories['wrapped_image'], self.memories['grid_change'], self.memories['rotation_pose']]
        self.editor.setInputMems(editor_inputs)
        editor_outputs = [self.memories['output_img']]
        self.editor.setOutputMems(editor_outputs)

