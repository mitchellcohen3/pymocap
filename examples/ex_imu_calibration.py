from pymocap import MocapTrajectory, IMUData
import rosbag

filename = "data/imu_calib.bag"
agent = "ifo002"

# Extract data
with rosbag.Bag(filename, "r") as bag:
    imu = IMUData.from_bag(bag, f"/{agent}/mavros/imu/data_raw")
    mocap = MocapTrajectory.from_bag(bag, agent)

# Do calibration. See function documentation for interpretation of the output.
C_bm, C_wl, gyro_bias, accel_bias = imu.calibrate(mocap)

# Apply calibration results to the data. 
imu = imu.apply_calibration(gyro_bias = gyro_bias, accel_bias = accel_bias)
mocap = mocap.rotate_body_frame(C_bm)
mocap = mocap.rotate_world_frame(C_wl)
# At this point, the data inside (mocap, imu) is calibrated and ready for use.

################################################################################
# We will test with some dead reckoning.
from pynav.lib.states import SE23State
from pynav.lib.imu import IMU, IMUKinematics
from typing import List
from pylie import SO3
import numpy as np
from tqdm import tqdm
np.set_printoptions(precision=3, suppress=True)
process = IMUKinematics(None)
imu_list: List[IMU] = imu.to_pynav()
traj_true: List[SE23State] = mocap.to_pynav(imu.stamps, extended_pose=True)
x = traj_true[0]

traj = [x]
print("Running dead reckoning...")
for k in tqdm(range(1, len(imu_list))):
    u = imu_list[k]
    dt = u.stamp - imu_list[k-1].stamp
    x = process.evaluate(x, u, dt)
    x.stamp = u.stamp
    traj.append(x)

t = np.array([x.stamp for x in traj])
att = np.array([SO3.Log(x.attitude).ravel() for x in traj])
att_true = np.array([SO3.Log(x.attitude).ravel() for x in traj_true])

vel = np.array([x.velocity for x in traj])
vel_true = np.array([x.velocity for x in traj_true])

pos = np.array([x.position for x in traj])
pos_true = np.array([x.position for x in traj_true])

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs[0].plot(t,att[:, 0], label="Estimated")
axs[0].plot(t,att_true[:, 0], label="True")
axs[1].plot(t,att[:, 1])
axs[1].plot(t,att_true[:, 1])
axs[2].plot(t,att[:, 2])
axs[2].plot(t,att_true[:, 2])
axs[0].legend()
axs[-1].set_xlabel("Time (s)")
axs[0].set_ylabel("Roll (rad)")
axs[1].set_ylabel("Pitch (rad)")
axs[2].set_ylabel("Yaw (rad)")
axs[0].set_ylim(-np.pi, np.pi)
axs[0].set_title("Attitude")

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs[0].plot(t,vel[:, 0], label="Estimated")
axs[0].plot(t,vel_true[:, 0], label="True")
axs[1].plot(t,vel[:, 1])
axs[1].plot(t,vel_true[:, 1])
axs[2].plot(t,vel[:, 2])
axs[2].plot(t,vel_true[:, 2])
axs[0].legend()
axs[-1].set_xlabel("Time (s)")
axs[0].set_ylabel("X (m/s)")
axs[1].set_ylabel("Y (m/s)")
axs[2].set_ylabel("Z (m/s)")
axs[0].set_title("Velocity")

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs[0].plot(t,pos[:, 0], label="Estimated")
axs[0].plot(t,pos_true[:, 0], label="True")
axs[1].plot(t,pos[:, 1])
axs[1].plot(t,pos_true[:, 1])
axs[2].plot(t,pos[:, 2])
axs[2].plot(t,pos_true[:, 2])
axs[0].legend()
axs[-1].set_xlabel("Time (s)")
axs[0].set_ylabel("X (m)")
axs[1].set_ylabel("Y (m)")
axs[2].set_ylabel("Z (m)")
axs[0].set_title("Position")
plt.show()
