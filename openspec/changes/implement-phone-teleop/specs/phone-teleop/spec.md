# Phone Teleoperation Spec

## ADDED Requirements

### Requirement: Remote Pose Streaming
The system SHALL accept external 6D pose streams to drive the robot's end-effector target.

#### Scenario: Startup Calibration
-   **Given** the user starts the `02_phone_ik.py` script.
-   **When** the first valid packet is received from the phone.
-   **Then** the system records this pose as `phone_initial_pose`.
-   **And** the robot target is set to the defined baseline `(0.3, 0.0, 0.2)` with identity rotation.

#### Scenario: Incremental Updates
-   **Given** the system has established `phone_initial_pose`.
-   **When** a new pose packet `phone_current_pose` arrives.
-   **Then** the system calculates the offset `delta = phone_current_pose - phone_initial_pose`.
-   **And** applies this offset to the robot baseline to determine the new `ik_target`.
-   **And** the visualizer updates the target marker location.
