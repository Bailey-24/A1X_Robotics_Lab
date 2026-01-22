# Implement Phone-Based Teleoperation

## Goal
Enable 6D pose control of the A1X robot in Viser simulation using streaming data from an iPhone via `asmagic`.

## User Review Required
-   **Calibration**: The specific axis mapping between phone and robot is not defined. We assume a direct mapping for now.
-   **Network**: Requires the iPhone and computer to be on the same network (IP hardcoded or configurable).

## Proposed Changes

### `pyroki/examples`

#### [NEW] `02_phone_ik.py`
-   Based on `01_basic_ik.py`.
-   Integrates `asmagic.ARDataSubscriber`.
-   Implements the incremental pose mapping logic.
-   Added thread for data reception.

## Verification Plan

### Manual Verification
-   **Simulated Motion**: Run `python pyroki/examples/02_phone_ik.py`.
-   **Phone Connection**: Ensure iPhone is streaming to the correct IP.
-   **Observation**:
    -   Verify that moving the phone moves the Viser target marker.
    -   Verify that the robot arm follows the marker inside the simulation.
    -   Verify that the initial position aligns with `(0.3, 0.0, 0.2)`.
