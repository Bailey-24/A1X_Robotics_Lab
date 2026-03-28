#!/usr/bin/env python3
"""律动舞蹈：joint1 左右摇摆 + joint2 点头 + joint4 腕部跟随，循环 3 轮。"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import a1x_control

HOME = [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]

LOOPS = 3       # 循环轮数
POINTS = 60     # 每轮轨迹点数
RATE_HZ = 10.0  # 执行频率


def make_dance_trajectory():
    """生成一轮舞蹈轨迹（60 个关节位姿）。"""
    points = []
    for i in range(POINTS):
        t = 2 * math.pi * i / POINTS  # 0 → 2π，一个完整周期
        j1 = 0.4 * math.sin(t)          # 底座左右摇摆
        j2 = 0.5 + 0.2 * math.sin(2*t)  # 肩部以 2 倍频点头
        j3 = -0.5                        # 肘部固定，保持手臂伸出
        j4 = 0.2 * math.cos(t)          # 腕部小幅跟随
        j5 = 0.0
        j6 = 0.0
        points.append([j1, j2, j3, j4, j5, j6])
    return points


def main():
    controller = a1x_control.JointController()
    time.sleep(2)

    if not controller.wait_for_joint_states(timeout=10):
        print("ERROR: 无法获取关节状态")
        return

    print("回到 home 位...")
    controller.move_to_position_smooth(HOME, steps=30, rate_hz=10.0, interpolation_type='cosine')
    time.sleep(1)

    trajectory = make_dance_trajectory()

    for loop in range(1, LOOPS + 1):
        print(f"第 {loop}/{LOOPS} 轮舞蹈...")
        controller.execute_trajectory(trajectory, rate_hz=RATE_HZ)

    print("舞蹈结束，回到 home 位...")
    controller.move_to_position_smooth(HOME, steps=30, rate_hz=10.0, interpolation_type='cosine')
    print("完成！")


if __name__ == "__main__":
    main()
