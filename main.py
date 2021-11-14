
import gym
import os

import mujoco_py
import numpy as np



def humanoid():
    mj_path = mujoco_py.utils.discover_mujoco()
    xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    viewer.render()

    print(sim.data.qpos)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    sim.step()
    viewer.render()
    print(sim.data.qpos)
    # [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
    #   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
    #   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
    #  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
    #  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
    #  -2.22862221e-05]

    while True:
        sim.step()
        viewer.render()

class HammerEnvV0():
    def __init__(self) -> None:
        self.setup_env()

    def setup_env(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mj_path = curr_dir + "/assets/DAPG_hammer.xml"
        model = mujoco_py.load_model_from_path(mj_path)
        self.model = model
        sim = mujoco_py.MjSim(model)
        self.sim = sim
        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.target_obj_sid = self.sim.model.site_name2id('S_target')
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.tool_sid = self.sim.model.site_name2id('tool')
        self.goal_sid = self.sim.model.site_name2id('nail_goal')
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

    def get_action(self):
        '''
        0 - arm left/right ARTx
        1 - arm up/down ARTy
        2 - arm forward/back ARTz
        3 - elbow down/up ARRx
        4 - elbow left/right ARRy
        5 - elbow twist ARRz
        6 - wrist up/down
        '''
        action = np.random.random((30,)) * 2 - 1
        action = np.clip(action, -1.0, 1.0)
        clip_movements = True
        if clip_movements:
            i = 6
            action[:i] = 0
            action[i+1:] = 0
            #action[6] = 0.5
            action[2] = -0.5

        #action = self.act_mid + action * self.act_rng # mean center and scale
        return action

    def run_hand_model(self):
        sim = self.sim
        viewer = mujoco_py.MjViewer(sim)
        viewer.render()

        while True:
            action = self.get_action()
            self.take_action(action)
            sim.step()

            # Rotate left/right
            viewer.cam.azimuth += 0.1
            viewer.render()

    def take_action(self, action):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = action[i]

def run_hand_model():
    hammer_env = HammerEnvV0()
    hammer_env.run_hand_model()

if __name__ == "__main__":
    run_hand_model()