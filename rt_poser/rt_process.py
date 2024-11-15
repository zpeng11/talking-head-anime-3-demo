import sys
import os
sys.path.append(os.getcwd())
from trt.trt_utils import *
from os.path import join
import torch
import numpy as np

class NodeInfo:
    def __init__(self, name:str, shape:List[int], dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
    def __str__(self) -> str:
        return self.name+": "+str(self.shape) + " "+str(self.dtype)
    def __repr__(self):
        return self.__str__()

class Proc:
    def __init__(self, path:str, n_inputs:int):
        TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {path}')
        runtime = trt.Runtime(TRT_LOGGER)
        with open(path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')

        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Creating inference context')
        self.context = self.engine.create_execution_context()
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed inference context')

        input_tensor_names = [self.engine.get_tensor_name(i) for i in range(n_inputs)]
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input nodes: '+ str(input_tensor_names))

        #self.in/output_tensor_info = List[item: Tuple(name: Str, shape: List[dim: Int])]
        self.input_tensor_info = []
        for name in input_tensor_names:
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.input_tensor_info.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.input_tensor_info))

        output_tensor_names = [self.engine.get_tensor_name(i) for i in range(n_inputs, self.engine.num_io_tensors)]
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output nodes: '+ str(output_tensor_names))

        self.output_tensor_info = []
        for name in output_tensor_names:
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = [dim for dim in self.context.get_tensor_shape(name)]
            info = NodeInfo(name, shape, dtype)
            self.output_tensor_info.append(info)
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input infos: '+ str(self.output_tensor_info))

    def setInputs(self, inputs:List[HostDeviceMem]): # inputs pass by reference
        assert(len(inputs) == len(self.input_tensor_names))
        for i in range(len(inputs)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.input_tensor_names[i])
            self.context.set_tensor_address(self.input_tensor_names[i], int(inputs[i].device)) # Use this setup without binding for v3
    def setOutputs(self, outputs:List[HostDeviceMem]): # outputs pass by reference
        assert(len(outputs) == len(self.output_tensor_names))
        for i in range(len(outputs)):
            TRT_LOGGER.log(TRT_LOGGER.INFO, 'Setting up for '+ self.output_tensor_names[i])
            self.context.set_tensor_address(self.output_tensor_names[i], int(outputs[i].device)) # Use this setup without binding for v3
    def exec(self, stream):
        self.context.execute_async_v3(stream.handle)

class RT_Process:
    def __init__(self, model_dir):
        self.prepareProcesses(model_dir)
        self.prepareMemories()
    def prepareProcesses(self, model_dir):
        self.decomposer = Proc(join(model_dir, 'decomposer.trt'))
        self.combiner = Proc(join(model_dir, 'combiner.trt'))
        self.morpher = Proc(join(model_dir, 'morpher.trt'))
        self.rotator = Proc(join(model_dir, 'rotator.trt'))
        self.editor = Proc(join(model_dir, 'editor.trt'))
    def prepareMemories(self):
        self.memories = {}
        self.memories['input_img'] = 