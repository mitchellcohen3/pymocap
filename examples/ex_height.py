# %%
from pymocap import MocapTrajectory, HeightData
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import rosbag

sns.set_theme(style="whitegrid")
filename = "data/imu_calib.bag"
agent = "ifo001"

# Extract data
with rosbag.Bag(filename) as bag:
    mocap = MocapTrajectory.from_bag(bag, agent)
    height = HeightData.from_bag(bag, f"/{agent}/mavros/distance_sensor/hrlv_ez4_pub")

# %% Plotting
height.plot(mocap)
plt.show()
