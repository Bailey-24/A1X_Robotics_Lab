"""Phone Teleoperation IK

Control the robot using 6D pose from an iPhone via AsMagic.
"""

import time
import threading
import copy
import numpy as np
import sys
import os

# Add local libraries to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../pyroki/src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../pyroki/examples"))

import pyroki as pk
import pyroki_snippets as pks
import viser
from viser.extras import ViserUrdf
from scipy.spatial.transform import Rotation as R
from asmagic import ARDataSubscriber
import yourdfpy
from pathlib import Path

# Hardcoded IP for now, matching user's example
PHONE_IP = "192.168.31.159"

class PhoneListener:
    def __init__(self, ip):
        self.ip = ip
        self.latest_pose = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        print(f"Connecting to phone at {self.ip}...")
        try:
            self.sub = ARDataSubscriber(self.ip)
            self.connected = True
            print("Phone connected.")
            for data in self.sub:
                if not self.running:
                    break
                if hasattr(data, 'local_pose') and data.local_pose is not None:
                    # Expecting [x, y, z, qx, qy, qz, qw]
                    pose = np.array(data.local_pose)
                    if pose.shape[0] == 7:
                        with self.lock:
                            self.latest_pose = pose
        except Exception as e:
            print(f"Phone listener error: {e}")
            self.connected = False
        finally:
            if hasattr(self, 'sub'):
                self.sub.close()

    def get_pose(self):
        with self.lock:
            return copy.deepcopy(self.latest_pose) if self.latest_pose is not None else None

    def stop(self):
        self.running = False


def main():
    print("Initializing Robot Simulation...")

    # Load Robot
    urdf_path = Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/lib/mobiman/configs/urdfs/a1x.urdf")
    
    def resolve_package_uri(fname: str) -> str:
        package_prefix = "package://mobiman/"
        if fname.startswith(package_prefix):
            relative_path = fname[len(package_prefix):]
            return str(Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/share/mobiman") / relative_path)
        return fname
    
    urdf = yourdfpy.URDF.load(urdf_path, filename_handler=resolve_package_uri)
    target_link_name = "gripper_link"
    robot = pk.Robot.from_urdf(urdf)

    # Setup Viser
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Initial Robot Target
    # Position: (0.3, 0.0, 0.2)
    # Rotation: Identity (wxyz=[1,0,0,0]) -> Scipy (xyzw=[0,0,0,1])
    init_target_pos = np.array([0.3, 0.0, 0.2])
    init_target_rot = R.from_quat([0, 0, 0, 1]) 
    
    # Visualizer control (disabled interaction, controlled by phone)
    ik_target = server.scene.add_transform_controls(
        "/ik_target", 
        scale=0.2, 
        position=init_target_pos, 
        wxyz=np.array([1, 0, 0, 0]),

    )

    # Start Phone Listener
    phone = PhoneListener(PHONE_IP)
    
    phone_base_pos = None
    phone_base_rot_inv = None
    
    print("Ready. Move phone to control robot.")

    while True:
        start_time = time.time()
        
        # 1. Get Phone Data
        phone_pose = phone.get_pose()
        
        target_pos = init_target_pos
        target_rot = init_target_rot

        if phone_pose is not None:
            # Parse [x, y, z, qx, qy, qz, qw]
            p_curr = phone_pose[:3]
            q_curr = phone_pose[3:] 
            
            # Normalize quaternion just in case
            norm = np.linalg.norm(q_curr)
            if norm > 0:
                q_curr = q_curr / norm
            
            r_curr = R.from_quat(q_curr)
            
            if phone_base_pos is None:
                # First frame: zeroing
                phone_base_pos = p_curr
                phone_base_rot_inv = r_curr.inv()
                print("Phone origin set.")
            
            # Calculate Deltas
            # Delta Position: relative to start
            delta_pos = p_curr - phone_base_pos
            
            # Delta Rotation: R_delta = R_start^-1 * R_curr
            delta_rot = phone_base_rot_inv * r_curr
            
            # Apply to Robot Initial Pose
            # P_target = P_init + Delta_P
            target_pos = init_target_pos + delta_pos
            
            # R_target = R_init * Delta_R
            target_rot = init_target_rot * delta_rot
            
            # Update Debug Visuals
            # Convert Scipy (xyzw) to Viser (wxyz)
            q_tgt = target_rot.as_quat()
            ik_target.position = target_pos
            ik_target.wxyz = np.array([q_tgt[3], q_tgt[0], q_tgt[1], q_tgt[2]])

        # 2. Solve IK
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )

        # 3. Update Robot Visual
        urdf_vis.update_cfg(solution)
        
        # Loop rate
        time.sleep(0.01)

if __name__ == "__main__":
    main()
