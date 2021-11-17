
import argparse
import os

import cv2
import gym
import mujoco_py
import numpy as np
import time

class ShapeXML():
    def __init__(self, name) -> None:
        self.name = name

class SphereXML(ShapeXML):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.x = 0
        self.y = 0
        self.z = 0
        self.size_x = 0.1
        self.size_y = 0.1
        self.size_z = 0.1
    def __str__(self):
        s = '<geom type="sphere" '
        s += 'name="{0}" '.format(self.name)
        s += 'pos="{0} {1} {2}" '.format(str(self.x), str(self.y), str(self.z))
        s += 'size="{0} {1} {2}" '.format(str(self.size_x), str(self.size_y), str(self.size_z))
        s += '/>'
        return s

class SimXML():
    def __init__(self) -> None:
        self.__shapes = {}

    def add_shape(self, shape:ShapeXML):
        self.__shapes[shape.name] = shape
    def __str__(self):
        s = '''
        <mujoco>
            <compiler angle="radian" />
            <worldbody>
                <body pos="0 0 0">
        '''
        for shape in self.__shapes.values():
            s += str(shape)
        s += '''
                </body>
                <camera name="camera_left" pos="-0.1 -1.2 1" fovy="90" axisangle="1 0 0 0.9"/>
                <camera name="camera_right" pos="0.1 -1.2 1" fovy="90" axisangle="1 0 0 0.9"/>
            </worldbody>
        </mujoco>
        '''
        return s

class SimGenerator():
    def __init__(self, img_shape) -> None:
        self.img_shape = img_shape
        self.left = np.random.randint(0, 255, size=self.img_shape, dtype=np.uint8)
        self.right = np.random.randint(0, 255, size=self.img_shape, dtype=np.uint8)


    def generate(self):
        sphere = SphereXML("sphere1")
        sim_xml = SimXML()
        sim_xml.add_shape(sphere)
        sim = mujoco_py.MjSim(mujoco_py.load_model_from_xml(str(sim_xml)))
        sim.step()
        im_left = sim.render(height=self.img_shape[0], width=self.img_shape[1], camera_name="camera_left")
        im_right = sim.render(height=self.img_shape[0], width=self.img_shape[1], camera_name="camera_right")
        self.left = im_left
        self.right = im_right

    def update(self, left_img, right_img):
        pass

def imagination(show_viewer, render):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(curr_dir, 'main.xml')
    model = mujoco_py.load_model_from_path(xml_path)
    real_world_sim = mujoco_py.MjSim(model)
    if show_viewer:
        viewer = mujoco_py.MjViewer(real_world_sim)
        viewer.render()

    # real_world_sim.data.ctrl[:] = 0.5
    width = 600
    height = 400
    sim_generator = SimGenerator((height, width, 3))

    while True:
        sim_generator.generate()
        real_world_sim.step()
        if show_viewer:
            viewer.render()
        else:
            img_left = real_world_sim.render(width=width, height=height, camera_name="camera_left")
            img_right = real_world_sim.render(width=width, height=height, camera_name="camera_right")
            sim_generator.update(img_left, img_right)
            if render:
                cv2.imshow("left", img_left)
                cv2.imshow("right", img_right)
                cv2.imshow("generated_left", sim_generator.left)
                cv2.imshow("generated_right", sim_generator.right)
                cv2.waitKey()
                return

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

    imagination(show_viewer, render)