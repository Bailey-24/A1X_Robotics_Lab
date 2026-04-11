import h5py

with h5py.File("data/demos/yoloe_grasp_white_object/0.hdf5", "r") as f:
    print("Groups:", list(f.keys()))
    # Groups: ['a1x_arm', 'cam_wrist']

    print("Arm fields:", list(f["a1x_arm"].keys()))
    # ['action', 'gripper', 'joint', 'timestamp']

    print("Joint shape:", f["a1x_arm/joint"].shape)        # (N, 6)
    print("Gripper shape:", f["a1x_arm/gripper"].shape)    # (N, 1)
    print("Action shape:", f["a1x_arm/action"].shape)      # (N, 7)
    print("Color shape:", f["cam_wrist/color"].shape)      # (N, 480, 640, 3)

    print("Joint :", f["a1x_arm/joint"])        # (N, 6)
    print("Gripper :", f["a1x_arm/gripper"])    # (N, 1)
    print("Action :", f["a1x_arm/action"])      # (N, 7)
    print("Color :", f["cam_wrist/color"])      # (N, 480, 640, 3)