# Phone-Based Teleoperation Design

## Problem
The user wants to control the A1X robot's end-effector in the Viser simulation using an iPhone's 6D pose data. The control should be relative to a fixed initial robot pose, mapping user hand motions directly to the robot's gripper.

## Architecture

### Components
1.  **Data Source**: `asmagic.ARDataSubscriber` connecting to the iPhone.
2.  **Controller**: A Python script (likely extending `pyroki/examples/01_basic_ik.py`) that:
    -   Reads phone pose.
    -   Computes relative transform.
    -   Updates the optimization target.
3.  **Solver**: `pyroki` IK solver (existing).
4.  **Visualizer**: `viser` (existing).

### Coordinate Mapping
-   **Robot Frame**: A1X base frame (z-up).
-   **Phone Frame**: ARKit/ARCore frame (y-up or z-up depending on initialization, usually initialized at app start).
-   **Logic**:
    -   On START: Record $T_{phone}^{init}$ and $T_{robot}^{init}$ (hardcoded to `pos=(0.3,0,0.2), rot=identity`).
    -   Loop:
        -   Read $T_{phone}^{curr}$.
        -   Compute $\delta T = (T_{phone}^{init})^{-1} \cdot T_{phone}^{curr}$ (relative motion in phone frame).
        -   Ideally, we want the user's hand motion to map intuitive to the robot. If the user moves hand "forward" (phone -z?), robot moves forward (+x?). We may need a reorientation matrix $T_{align}$.
        -   Simple incremental approach:
            -   $P_{robot}^{target} = P_{robot}^{init} + (R_{align} \cdot (P_{phone}^{curr} - P_{phone}^{init}))$.
            -   $R_{robot}^{target} = R_{robot}^{init} \cdot (R_{align} \cdot (R_{phone}^{init})^{-1} \cdot R_{phone}^{curr})$. (Simplified, rotation mapping might need tuning).
    -   For now, we will implement a direct 1:1 mapping of increments, possibly remapping axes if needed (e.g., phone Z -> robot X). *Self-correction: User didn't ask for axis remapping, just "incremental changes". I will stick to direct mapping first.*

### Concurrency
-   `ARDataSubscriber` is a generator. We can't block the `viser` / basic IK loop.
-   We will likely need a separate thread to consume `ARDataSubscriber` and update a shared state `current_phone_pose`.
