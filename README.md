# SMPLH_To_3D_Joints

- run.py
사용법)
cli상에서 요구하는 파라미터.
-- input_path: .npz의 smplh 경로
-- output_path: 처리된 3차원 joints (Frames, 22, 3)
-- body_path: 손댈 필요 없음.

예시)
$ python run.py --input_path {path}/smplh_file.npy --output_path {path}/{name}.npy
