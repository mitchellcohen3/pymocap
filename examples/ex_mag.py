# %%
from pymocap import MocapTrajectory, MagnetometerData
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import rosbag

sns.set_theme(style="whitegrid")
filename = "data/imu_calib.bag"
agent = "ifo002"

# Extract data
with rosbag.Bag(filename) as bag:
    mocap = MocapTrajectory.from_bag(bag, agent)
    mag = MagnetometerData.from_bag(bag, f"/{agent}/mavros/imu/mag")

D, b, m_a = mag.calibrate(mocap)
mag_calib = mag.apply_calibration(D, b)
# %% Plotting
fig, ax = plt.subplots(1, 1)
m = mag.magnetic_field
ax.scatter(m[:, 0], m[:, 1], label="x vs y")
ax.scatter(m[:, 1], m[:, 2], label="y vs z")
ax.scatter(m[:, 0], m[:, 2], label="x vs z")
ax.legend()
ax.set_title("Raw data")
ax.axis("equal")

fig, ax = mag_calib.plot(mocap, mag_vector=m_a)
fig, ax = plt.subplots(1, 1)
m = mag_calib.magnetic_field
ax.scatter(m[:, 0], m[:, 1], label="x vs y")
ax.scatter(m[:, 1], m[:, 2], label="y vs z")
ax.scatter(m[:, 0], m[:, 2], label="x vs z")
ax.axis("equal")
ax.legend()
ax.set_title("Calibrated")
plt.show()
