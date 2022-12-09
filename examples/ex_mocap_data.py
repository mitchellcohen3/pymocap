# %%
from pymocap import MocapTrajectory
from typing import List 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
filename = "data/imu_calib.bag"
agent = "ifo001"

# Extract data
mocap = MocapTrajectory.from_bag(filename, agent)
pos = mocap.position(mocap.stamps)
quat = mocap.quaternion(mocap.stamps)
vel = mocap.velocity(mocap.stamps)
accel = mocap.acceleration(mocap.stamps)
imu_accel = mocap.accelerometer(mocap.stamps)
omega = mocap.angular_velocity(mocap.stamps)
is_static = mocap.is_static(mocap.stamps)

# %% Plotting
################################################################################
### POSITION
fig, axs = plt.subplots(3, 1, sharex=True)
axs: List[plt.Axes] = axs
pos = mocap.position(mocap.stamps)
axs[0].plot(mocap.stamps, pos[:, 0])
axs[1].plot(mocap.stamps, pos[:, 1])
axs[2].plot(mocap.stamps, pos[:, 2])
axs[2].plot(mocap.stamps, is_static.astype(int), label="Static")
axs[0].set_title("Mocap Position Trajectory")
axs[2].legend()

################################################################################
### VELOCITY
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs: List[plt.Axes] = axs
axs[0].plot(mocap.stamps, vel[:, 0])
axs[1].plot(mocap.stamps, vel[:, 1])
axs[2].plot(mocap.stamps, vel[:, 2])
axs[0].set_title("Mocap Velocity Trajectory")

################################################################################
### ACCELERATION
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs: List[plt.Axes] = axs
axs[0].plot(mocap.stamps, accel[:, 0])
axs[1].plot(mocap.stamps, accel[:, 1])
axs[2].plot(mocap.stamps, accel[:, 2])
axs[0].set_title("Mocap Acceleration Trajectory")


################################################################################
### QUATERNION
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
axs: List[plt.Axes] = axs
axs[0].plot(mocap.stamps, quat[:, 0])
axs[1].plot(mocap.stamps, quat[:, 1])
axs[2].plot(mocap.stamps, quat[:, 2])
axs[3].plot(mocap.stamps, quat[:, 3])
axs[0].set_title("Mocap Quaternion Trajectory")
axs[0].set_ylim(-1.05, 1.05)

################################################################################
### ANGULAR VELOCITY
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs: List[plt.Axes] = axs
axs[0].plot(mocap.stamps, omega[:, 0])
axs[1].plot(mocap.stamps, omega[:, 1])
axs[2].plot(mocap.stamps, omega[:, 2])
axs[0].set_title("Mocap Angular Velocity Trajectory")
axs[0].set_ylim(-1.05, 1.05)


plt.show()
