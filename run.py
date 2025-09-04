import argparse
import torch

import os, sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "human_body_prior", "src"))

from human_body_prior.body_model.body_model import BodyModel
import numpy as np
from os import path as osp

from colorama import Fore, Style, init
init(autoreset=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
betas = 0

def load_Body_Model(bm_path, num_betas=16):
    bm = BodyModel(
        bm_fname=bm_path,      # <- 이름 맞추기
        num_betas=num_betas,   # 10 또는 16 권장
        num_dmpls=None,
        dmpl_fname=None,
        num_expressions=None,  # smplh에선 무시됨
        use_posedirs=True,
        model_type="smplh",    # <- 키워드로 지정 (중요)
        dtype=torch.float32,
        persistant_buffer=False
    ).to(device)
    bm.eval()
    return bm

@torch.no_grad()
def smplh_joints_from_npz(npz_path, bm, root_relative=False, body22_only=True):  # True면 손가락 제외 앞 22관절만
    bdata = np.load(npz_path, allow_pickle=True)
    T = len(bdata["trans"])

    # AMASS(SMPL-H) 포즈 분할: 3(root) + 63(body) + 90(hands) = 156
    root_orient = torch.from_numpy(bdata["poses"][:, :3]).float().to(device)
    pose_body   = torch.from_numpy(bdata["poses"][:, 3:66]).float().to(device)

    # 핸드 포즈 없으면 0으로
    if bdata["poses"].shape[1] >= 156:
        pose_hand = torch.from_numpy(bdata["poses"][:, 66:156]).float().to(device)
    else:
        pose_hand = torch.zeros((T, 90), dtype=torch.float32, device=device)

    trans = torch.from_numpy(bdata["trans"]).float().to(device)

    # 평균 체형이 목적이면 betas는 아예 전달하지 않거나 0으로 전달
    # (num_betas=16 기준)
    betas = torch.zeros((T, bm.num_betas), dtype=torch.float32, device=device)

    out = bm(root_orient=root_orient,
             pose_body=pose_body,
             pose_hand=pose_hand,
             trans=trans,
             betas=betas)

    joints3d = out.Jtr  # (T, 52, 3), 22 + 손관절 => 52
    if root_relative:
        joints3d = joints3d - joints3d[:, :1, :]

    if body22_only:
        joints3d = joints3d[:, :22, :]

    return joints3d.cpu().numpy()
    
def main():
    parser = argparse.ArgumentParser(description="get 3D Joints from SMPLH")
    parser.add_argument("--input_path", type=str, required=True, help='.npz format')
    parser.add_argument("--output_path", type=str, required=True, help='.npy format')
    parser.add_argument("--body_path", type=str, default='./smplh_model/male/model.npz')
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    body_path = args.body_path
    
    body_model = load_Body_Model(bm_path=body_path)
    out_npy = smplh_joints_from_npz(input_path, body_model)
    
    print(Fore.GREEN + f'out_npy_shape: {out_npy.shape}')

    np.save(output_path, out_npy)
    print(Fore.GREEN + f'Save Done: {output_path}')    
    
if __name__ == "__main__":
    main()