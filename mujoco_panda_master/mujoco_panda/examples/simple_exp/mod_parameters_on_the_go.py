import time
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

XML = '''
<mujoco>
    <worldbody>
        <geom name='floor' pos='0 0 0' size='5 5 .125' type='plane' condim='3'/>
        <body name='ball' pos='0 0 1'>
            <joint type='free'/>
            <geom name='ball' pos='0 0 0' size='.1' type='sphere' rgba='1 0 0 1'/>
        </body>
    </worldbody>
</mujoco>
'''


model = load_model_from_xml(XML)
sim = MjSim(model)
viewer = MjViewer(sim)

while True:
    sim.model.opt.gravity[0] = np.sin(time.time())
    sim.model.opt.gravity[1] = np.cos(time.time())
    sim.step()
    viewer.render()