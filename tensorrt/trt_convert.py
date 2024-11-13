from trt_utils import *


model_dir = './fp16/'
dtype = np.float16

decomposer_engine = load_engine(join(model_dir, 'decomposer.trt'))
decomposer_proc = Processor(decomposer_engine, 4)

combiner_engine = load_engine(join(model_dir, 'combiner.trt'))
combiner_proc = Processor(combiner_engine, 4)

morpher_engine = load_engine(join(model_dir, 'morpher.trt'))
morpher_proc = Processor(morpher_engine, 4)
morpher_proc.loadInputs([np.random.rand(*shape).astype(dtype) for shape in morpher_input_shapes])

rotator_engine = load_engine(join(model_dir, 'rotator.trt'))
rotator_proc = Processor(rotator_engine, 2)
rotator_proc.loadInputs([np.random.rand(*shape).astype(dtype) for shape in rotator_input_shapes])

tmp = [np.random.rand(*shape).astype(dtype) for shape in rotator_input_shapes]

editor_engine = load_engine(join(model_dir, 'editor.trt'))
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