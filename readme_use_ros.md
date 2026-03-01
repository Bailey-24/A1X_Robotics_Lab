

# A1X sdk
doc: https://docs.galaxea-dynamics.com/Guide/A1XY/software_introduction/ROS2/A1XY_Software_Introduction_ROS2/#gripper-control


## setup can 
Note: Before starting nodes, please confirm that the CAN driver has been properly configured.
```bash
sudo ip link set can0 type can bitrate 1000000 sample-point 0.875 dbitrate 5000000 fd on dsample-point 0.875
sudo ip link set up can0
```
If you encounter the error "RTNETLINK answers: Device or resource busy", it usually means that the device you are trying to configure (such as a network interface or CAN transceiver) has already been configured and is running.   


## API control
```bash
conda activate a1x_ros
cd example
python motion/joint_control_once.py
python motion/joint_control_smooth.py
python motion/gripper_control.py
python ik_control_viser.py
```


##  joint control
cd /home/ubuntu/projects/A1Xsdk

### terminal 0 for launch driver
```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.zsh  
ros2 launch HDAS a1xy.py
```

### terminal 1 for launch rivi and some
```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.zsh  
ros2 launch mobiman A1x_jointTrackerdemo_launch.py
```

### terminal 2 for control joint 
```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.zsh  
ros2 topic echo /joint_states       # see the joint state

ros2 topic pub /motion_target/target_joint_state_arm sensor_msgs/msg/JointState "
name:
- arm_joint1
- arm_joint2
- arm_joint3
- arm_joint4
- arm_joint5
- arm_joint6
position:
- 0.0
- 0.0043
- -0.1 
- -0.0347
- -0.0055
- 0.0013
"
```


## ik control

### terminal 0 for launch driver
```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.zsh  
ros2 launch HDAS a1xy.py
```

### terminal 1
```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.zsh  
ros2 launch mobiman A1x_arm_relaxed_ik_launch.py
```
记得conda deactivate,和退出joint control的 launch

### terminal 2
```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.zsh  
ros2 topic echo /hdas/pose_ee_arm # see the end effector state

ros2 topic pub --once /relaxed_ik/pose_ee_arm geometry_msgs/msg/PoseStamped "
header:
  frame_id: base_link
pose:
  position:
    x: 0.00037
    y: -0.0587
    z: 0.2371
  orientation:
    x: -0.0115
    y: -0.0127
    z: -0.7031
    w: 0.7108
"
```


## gripper control
### terminal 0 for launch driver
```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.zsh  
ros2 launch HDAS a1xy.py
```

## terminal 1 for gripper setup
```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.zsh
ros2 launch mobiman A1xy_gripperController_launch.py
```

### terminal 2
```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.zsh 
ros2 topic echo /hdas/feedback_gripper # see the gripper state, 0 is close, 100 is open

ros2 topic pub --once /motion_target/target_position_gripper sensor_msgs/msg/JointState "
position: [100.0]
"
```


