import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from colorama import Fore, Style, init
init(autoreset=True)
import numpy as np

def vis_video(body_joints, output_path):
    T = body_joints.shape[0]
    
    bones = [
    (0,3), (3,6), (6,9), (9,15),            # spine to head
    (0,1), (1,4), (4,7), (7,10),           # 왼쪽 다리
    (0,2), (2,5), (5,8), (8,11),           # 오른쪽 다리
    (9,12), (12,16), (16,18), (18,20),     # 왼쪽 팔
    (9,12), (12,17), (17,19), (19,21),     # 오른쪽 팔
]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.view_init(elev=30, azim=-60)

    scat = ax.scatter([], [], [], c='r', s=20)
    lines = [ax.plot([], [], [], c='k', linewidth=2)[0] for _ in bones]
    title = ax.set_title("")

    def init():
        scat._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        title.set_text("Frame 0")
        return [scat, *lines, title]

    def update(frame):
        joints = body_joints[frame]  # (22,3)
        x, y, z = joints[:,0], joints[:,1], joints[:,2]
        scat._offsets3d = (x, y, z)
        for i, (a,b) in enumerate(bones):
            xa, ya, za = joints[a]
            xb, yb, zb = joints[b]
            lines[i].set_data([xa, xb], [ya, yb])
            lines[i].set_3d_properties([za, zb])
        title.set_text(f"Frame {frame}/{T}")
        return [scat, *lines, title]

    anim = FuncAnimation(fig, update, frames=range(0, T, 4), init_func=init,
                        interval=30, blit=False, repeat=False)
    #print("Elevation (ax.elev):", ax.elev)
    #print("Azimuth (ax.azim):", ax.azim)
    
    #plt.show()

    anim.save(output_path, dpi=150, fps=30)

def main():
    parser = argparse.ArgumentParser(description="vis smplh 3d joints")
    parser.add_argument("--input_path", type=str, required=True, help='.npy')
    parser.add_argument("--output_path", type=str, required=True, help='.video')
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    
    
    body_joints = np.load(input_path)
    print(Fore.GREEN + f'Load npy: {body_joints.shape}')
    
    vis_video(body_joints, output_path)

if __name__=='__main__':
    main()