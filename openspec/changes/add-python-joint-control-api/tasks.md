## 1. Unified API Implementation
- [x] 1.1 Create `a1x_control.py` module with automatic system initialization
- [x] 1.2 Implement auto-startup of HDAS driver and mobiman stack on import
- [x] 1.3 Add JointController class with simple method interface
- [x] 1.4 Add joint state reading functionality (subscribe to `/joint_states`)
- [x] 1.5 Add joint position command functionality (publish to `/motion_target/target_joint_state_arm`)
- [x] 1.6 Implement background process management and cleanup

## 3. User Interface and Documentation
- [x] 3.1 Create simple example scripts demonstrating API usage
- [x] 3.2 Add comprehensive docstrings and type hints
- [x] 3.3 Create README with setup and usage instructions
- [x] 3.4 Add validation for joint limits and safety constraints

## 4. Integration and Testing
- [x] 4.1 Test API with existing ROS stack
- [x] 4.2 Verify compatibility with HDAS driver and mobiman packages
- [x] 4.3 Add graceful shutdown and cleanup procedures
- [x] 4.4 Test error scenarios and recovery mechanisms
