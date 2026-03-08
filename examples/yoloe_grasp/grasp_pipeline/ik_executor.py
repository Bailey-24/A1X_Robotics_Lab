#!/usr/bin/env python3
"""IK & Motion Execution Module for Grasp Pipeline.

Solves inverse kinematics using PyRoki and executes smooth multi-phase
grasp motion via a1x_control.

Phases: pre-grasp → grasp → close gripper → lift

Usage (standalone test):
    python examples/yoloe_grasp/grasp_pipeline/ik_executor.py
"""
from __future__ import annotations

import sys
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class IKExecutor:
    """Solves IK and executes grasp motion sequences on the A1X robot."""

    def __init__(
        self,
        smooth_steps: int = 30,
        control_rate_hz: float = 10.0,
        joint_limits_min: Optional[List[float]] = None,
        joint_limits_max: Optional[List[float]] = None,
        interpolation_type: str = 'linear',
    ):
        """Initialize the IK executor.

        Args:
            smooth_steps: Number of interpolation steps for smooth motion.
            control_rate_hz: Control loop rate in Hz.
            joint_limits_min: Minimum joint limits in radians (6 joints).
            joint_limits_max: Maximum joint limits in radians (6 joints).
            interpolation_type: 'linear' or 'cosine' for trajectory generation.
        """
        self.smooth_steps = smooth_steps
        self.control_rate_hz = control_rate_hz
        self.interpolation_type = interpolation_type
        # Defaults match the A1X URDF exactly
        self.joint_limits_min = joint_limits_min or [-2.8798, 0.0, -3.3161, -1.5708, -1.5708, -2.8798]
        self.joint_limits_max = joint_limits_max or [ 2.8798, 3.1416, 0.0,  1.5708,  1.5708,  2.8798]

        # Load URDF and create PyRoki robot
        self.robot, self.target_link_name = self._load_robot()

    def _load_robot(self):
        """Load the A1X URDF and create a PyRoki robot."""
        pyroki_examples = str(Path(__file__).parent.parent.parent.parent / "pyroki" / "examples")
        if pyroki_examples not in sys.path:
            sys.path.insert(0, pyroki_examples)

        import pyroki as pk
        import yourdfpy

        urdf_path = Path(
            "/home/ubuntu/projects/A1Xsdk/install/mobiman/lib/mobiman/configs/urdfs/a1x.urdf"
        )

        def resolve_package_uri(fname: str) -> str:
            package_prefix = "package://mobiman/"
            if fname.startswith(package_prefix):
                relative_path = fname[len(package_prefix):]
                return str(
                    Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/share/mobiman")
                    / relative_path
                )
            return fname

        urdf = yourdfpy.URDF.load(urdf_path, filename_handler=resolve_package_uri)
        robot = pk.Robot.from_urdf(urdf)
        target_link_name = "gripper_link"

        logger.info(f"Loaded robot with {robot.joints.num_actuated_joints} actuated joints")
        return robot, target_link_name

    def solve_ik(
        self,
        target_position: np.ndarray,
        target_wxyz: np.ndarray,
        initial_joints: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Solve IK for a target end-effector pose (6 arm joints only).

        The URDF has 8 actuated joints (6 arm + 2 gripper), but only the
        6 arm joints participate in IK.  Gripper joints are pinned to 0.

        Args:
            target_position: Target position [x, y, z] in base frame.
            target_wxyz: Target orientation as quaternion [w, x, y, z].
            initial_joints: Seed config — either 6 arm joints or 8 full joints.
                            If 8 values, gripper joints (indices 6,7) are ignored.

        Returns:
            Joint angles (8 values: 6 arm + 2 gripper fixed at 0), or None.
        """
        pyroki_examples = str(Path(__file__).parent.parent.parent.parent / "pyroki" / "examples")
        if pyroki_examples not in sys.path:
            sys.path.insert(0, pyroki_examples)

        import pyroki_snippets as pks

        # Build 8-joint seed with gripper pinned to 0
        if initial_joints is None:
            seed8 = np.zeros(8)
        elif len(initial_joints) == 6:
            seed8 = np.zeros(8)
            seed8[:6] = initial_joints
        else:
            seed8 = initial_joints.copy()
            seed8[6:] = 0.0  # always pin gripper

        # Generate diverse seeds: given + observation + random within URDF limits
        lo = np.array(self.joint_limits_min)
        hi = np.array(self.joint_limits_max)
        num_random = 10

        seeds = [seed8.copy()]
        # Observation pose as second seed
        obs_seed = np.array([0.0, 1.0, -0.93, 0.83, 0.0, 0.0, 0.0, 0.0])
        seeds.append(obs_seed)
        # Random seeds sampled within URDF arm joint limits
        for _ in range(num_random):
            rng = np.zeros(8)
            rng[:6] = lo + (hi - lo) * np.random.rand(6)
            seeds.append(rng)

        target_idx = self.robot.links.names.index(self.target_link_name)
        best_solution = None
        best_error = float("inf")
        best_in_limits = False

        for i, s in enumerate(seeds):
            try:
                solution = pks.solve_ik(
                    robot=self.robot,
                    target_link_name=self.target_link_name,
                    target_position=target_position,
                    target_wxyz=target_wxyz,
                    initial_joint_config=s,
                )
                solution[6:] = 0.0  # pin gripper

                # Check arm joint limits
                arm_joints = solution[:6]
                in_limits = all(
                    lo_j <= v <= hi_j
                    for v, lo_j, hi_j in zip(
                        arm_joints, self.joint_limits_min, self.joint_limits_max
                    )
                )

                # Compute FK error
                fk_result = self.robot.forward_kinematics(solution)
                ee_pos = np.array(fk_result[target_idx][4:7])
                error = np.linalg.norm(ee_pos - target_position)

                if error < 0.01:
                    logger.info(
                        f"Seed {i}: FK error={error:.4f}m ✓  "
                        f"joints={arm_joints.tolist()}"
                    )

                # Prefer in-limits solutions, then lowest error
                better = False
                if in_limits and (not best_in_limits or error < best_error):
                    better = True
                elif not best_in_limits and error < best_error:
                    better = True

                if better:
                    best_error = error
                    best_solution = solution
                    best_in_limits = in_limits

                # Early exit if excellent solution
                if error < 0.005 and in_limits:
                    break

            except Exception:
                pass

        if best_solution is not None:
            logger.info(
                f"Best IK: error={best_error:.4f}m, "
                f"in_limits={best_in_limits}, "
                f"joints={best_solution[:6].tolist()}"
            )
            return best_solution
        else:
            logger.error("IK solve failed for all seeds")
            return None

    def verify_ik_solution(
        self,
        solution: np.ndarray,
        target_position: np.ndarray,
        position_tolerance: float = 0.01,
    ) -> bool:
        """Verify IK solution reaches the target by computing FK.

        Args:
            solution: Joint angles from IK solve (8 values).
            target_position: Target position [x, y, z].
            position_tolerance: Maximum allowed position error (meters).

        Returns:
            True if FK result is close enough to target.
        """
        target_idx = self.robot.links.names.index(self.target_link_name)
        fk_result = self.robot.forward_kinematics(solution)
        ee_pose = fk_result[target_idx]  # [wxyz(4), xyz(3)]
        ee_pos = np.array(ee_pose[4:7])

        error = np.linalg.norm(ee_pos - target_position)
        ok = error < position_tolerance

        logger.info(
            f"IK verification: target={target_position}, "
            f"achieved={ee_pos}, error={error:.4f}m, "
            f"{'PASS' if ok else 'FAIL'}"
        )
        return ok

    def execute_grasp_sequence(
        self,
        controller,
        T_pre_grasp_base: np.ndarray,
        T_grasp_base: np.ndarray,
        T_lift_base: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        dry_run: bool = False,
        confirm_each_phase: bool = False,
        gripper_close_delay: float = 2.5,
    ) -> bool:
        """Execute the full grasp motion sequence.

        Phases:
            1. Move to pre-grasp pose
            2. Move to grasp pose
            3. Close gripper
            4. Lift

        Args:
            controller: a1x_control.JointController instance.
            T_pre_grasp_base: 4x4 pre-grasp pose in base frame.
            T_grasp_base: 4x4 grasp pose in base frame.
            T_lift_base: 4x4 lift pose in base frame.
            current_joints: Current joint angles (8 values) as IK seed.
            dry_run: If True, only compute and print — do not move.
            confirm_each_phase: If True, prompt user before each motion.

        Returns:
            True if all phases completed successfully.
        """
        from .coordinate_transform import matrix_to_position_wxyz

        phases = [
            ("Pre-grasp", T_pre_grasp_base),
            ("Grasp", T_grasp_base),
        ]

        seed = current_joints if current_joints is not None else np.zeros(8)

        for phase_name, T_target in phases:
            pos, wxyz = matrix_to_position_wxyz(T_target)
            logger.info(f"\n--- Phase: {phase_name} ---")
            logger.info(f"  Target position: {pos}")
            logger.info(f"  Target orientation (wxyz): {wxyz}")

            # Solve IK
            solution = self.solve_ik(pos, wxyz, initial_joints=seed)
            if solution is None:
                logger.error(f"IK failed for {phase_name} phase")
                return False

            # Verify
            if not self.verify_ik_solution(solution, pos, position_tolerance=0.02):
                logger.error(f"IK verification FAILED for {phase_name} — "
                             f"aborting to prevent unsafe motion")
                return False

            arm_joints = list(solution[:6].astype(float))
            logger.info(f"  Joint targets: {arm_joints}")

            if dry_run:
                logger.info(f"  [DRY RUN] Skipping motion for {phase_name}")
                seed = solution
                continue

            if confirm_each_phase:
                response = input(f"  Execute {phase_name}? (y/n): ")
                if response.lower() != "y":
                    logger.info("  Aborted by user")
                    return False

            # Execute motion
            logger.info(f"  Moving to {phase_name} pose...")
            if phase_name == "Pre-grasp":
                # Pre-grasp: larger jump from observation, use more steps for
                # a slower/gentler motion the motors can track
                success = controller.move_to_position_smooth(
                    arm_joints,
                    steps=self.smooth_steps * 2,  # 60 steps → slower
                    rate_hz=self.control_rate_hz,
                    interpolation_type=self.interpolation_type,
                    wait_for_convergence=True,
                )
            else:
                # Grasp: small descent from pre-grasp
                success = controller.move_to_position_smooth(
                    arm_joints,
                    steps=self.smooth_steps,
                    rate_hz=self.control_rate_hz,
                    interpolation_type=self.interpolation_type,
                    wait_for_convergence=True,
                )
            if not success:
                logger.error(f"  Motion failed for {phase_name}")
                return False
            time.sleep(0.5)
            seed = solution

        # --- Close gripper ---
        logger.info("\n--- Phase: Close Gripper ---")
        if dry_run:
            logger.info("  [DRY RUN] Skipping gripper close")
        else:
            if confirm_each_phase:
                response = input("  Close gripper? (y/n): ")
                if response.lower() != "y":
                    logger.info("  Aborted by user")
                    return False

            logger.info("  Closing gripper...")
            controller.close_gripper()
            time.sleep(0.5)
            controller.close_gripper()  # send again for reliability
            time.sleep(gripper_close_delay)

        # --- Lift ---
        logger.info("\n--- Phase: Lift ---")
        pos_lift, wxyz_lift = matrix_to_position_wxyz(T_lift_base)
        logger.info(f"  Lift position: {pos_lift}")

        solution_lift = self.solve_ik(pos_lift, wxyz_lift, initial_joints=seed)
        if solution_lift is None:
            logger.error("IK failed for lift phase")
            return False

        arm_joints_lift = list(solution_lift[:6].astype(float))
        logger.info(f"  Joint targets: {arm_joints_lift}")

        if dry_run:
            logger.info("  [DRY RUN] Skipping lift motion")
        else:
            if confirm_each_phase:
                response = input("  Execute lift? (y/n): ")
                if response.lower() != "y":
                    logger.info("  Aborted by user")
                    return False

            logger.info("  Lifting...")
            success = controller.move_to_position_smooth(
                arm_joints_lift,
                steps=self.smooth_steps * 2,  # slower lift
                rate_hz=self.control_rate_hz,
                interpolation_type=self.interpolation_type,
                wait_for_convergence=False, # no need to strictly check convergence at the end of lift
            )
            if not success:
                logger.error("  Lift motion failed")
                return False

        logger.info("\n=== Grasp sequence complete! ===")
        return True


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("IK Executor Test")
    print("=" * 40)

    executor = IKExecutor(smooth_steps=30, control_rate_hz=10.0)

    # Test IK for a known reachable pose
    target_pos = np.array([0.25, 0.0, 0.2])
    target_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation

    print(f"\nTarget position: {target_pos}")
    print(f"Target orientation: {target_wxyz}")

    solution = executor.solve_ik(target_pos, target_wxyz)
    if solution is not None:
        print(f"Solution (arm joints): {solution[:6].tolist()}")
        executor.verify_ik_solution(solution, target_pos)
    else:
        print("IK solve failed!")
