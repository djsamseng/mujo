
import argparse
import os

import cv2
import gym
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
    def __init__(self, show_viewer, render) -> None:
        self.show_viewer = show_viewer
        self.render = render
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
        6 - write left/right
        7 - wrist up/down
        '''
        action = np.random.random((30,)) * 2 - 1
        action = np.clip(action, -1.0, 1.0)
        clip_movements = False
        if clip_movements:
            i = 6
            action[:i] = 0
            action[i+1:] = 0
            #action[6] = 0.5
            action[2] = -0.5

        #action = self.act_mid + action * self.act_rng # mean center and scale
        return action

    def get_actuator_names(self):
        actuator_addresses = self.model.name_actuatoradr
        return self.get_names_from_addresses(actuator_addresses)

    def get_names_from_addresses(self, addresses):
        names = self.model.names[
            addresses[0]:addresses[-1]
        ].tolist()
        for i in range(addresses[-1], addresses[-1] + 100):
            val = self.model.names[i]
            if val == b'':
                break
            else:
                names.append(val)
        
        names = [c.decode() for c in names]
        
        final_names = []
        cur_name = ""
        for a in names:
            if a == '':
                final_names.append(cur_name)
                cur_name = ""
            else:
                cur_name += a
        
        return final_names

    def get_body_names(self):
        body_addresses = self.model.name_bodyadr
        return self.get_names_from_addresses(body_addresses)

    def run_hand_model(self):
        sim = self.sim
        if self.show_viewer:
            viewer = mujoco_py.MjViewer(sim)
            viewer.render()

        print(self.model.actuator_ctrllimited)
        print(self.model.actuator_ctrlrange)
        actuator_names = self.get_actuator_names()
        print(actuator_names)
        body_names = self.get_body_names()
        print(body_names, len(body_names))
        print(self.model.body_pos.shape)
        
        last_pos = self.model.body_pos.copy()
        while True:
            action = self.get_action()
            self.take_action(action)
            sim.step()
            #print(self.sim.data.get_body_xpos("forearm"))
            np.testing.assert_allclose(last_pos, self.model.body_pos)

            # Rotate left/right
            if self.show_viewer:
                viewer.cam.azimuth += 0.1
                viewer.render()
            else:
                ret = sim.render(width=600, height=400, camera_name="forearm_camera")
                if self.render:
                    rgb_im = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
                    rgb_im = cv2.flip(rgb_im, 0)
                    cv2.imshow("Forearm camera", rgb_im)
                    cv2.waitKey()
                    return

    def take_action(self, action):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = action[i]

def run_hand_model(show_viewer, render):
    hammer_env = HammerEnvV0(show_viewer, render)
    hammer_env.run_hand_model()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", dest="viewer", nargs="?", default=False, const=True)
    parser.add_argument("--render", dest="render", nargs="?", default=False, const=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    show_viewer = args.viewer
    render = not show_viewer and args.render
    print("viewer:", show_viewer, "render:", render)

    run_hand_model(show_viewer, render)