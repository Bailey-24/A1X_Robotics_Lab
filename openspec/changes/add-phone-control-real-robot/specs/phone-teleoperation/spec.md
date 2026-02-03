# Phone Teleoperation Capability

## ADDED Requirements

### Requirement: Script SHALL support phone pose input mode
The IK control script SHALL support phone-based teleoperation as an alternative input source for the end-effector target.

#### Scenario: User enables phone control
- **WHEN** the user enables the "Phone Control" checkbox and the phone sends pose data
- **THEN** the IK target SHALL follow the phone's relative movement from its captured origin

#### Scenario: User switches back to manual mode
- **WHEN** the user disables the "Phone Control" checkbox
- **THEN** the IK target SHALL remain at its current position and become gizmo-controllable again

### Requirement: User SHALL be able to reset phone origin
The system SHALL allow the user to re-zero the phone origin at any time during phone control.

#### Scenario: User resets phone origin
- **WHEN** the user clicks "Reset Phone Origin"
- **THEN** the phone origin SHALL be recaptured from the next phone frame and the target SHALL return to the initial position

### Requirement: UI SHALL display phone connection status
The UI SHALL display the current phone connection status.

#### Scenario: Phone connects successfully
- **WHEN** the phone establishes connection
- **THEN** the phone status display SHALL show "Connected"

#### Scenario: Phone disconnects
- **WHEN** the phone connection is lost
- **THEN** the phone status display SHALL show "Disconnected" and target SHALL freeze at last position
