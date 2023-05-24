import pickle
from pymocap import MocapTrajectory
import numpy as np 

filename= "data/2023-05-08-T16-56_gt.pkl"

# `data` is a pandas dataframe with the index being the DatetimeIndex.
# The columns correponding to x,y,z position coordinates are 
# x_trans, y_trans, z_trans. The columns corresponding to the
# quaternion are x_rot, y_zot, z_rot, w.
data = pickle.load(open(filename, "rb"))

# Convert indices to seconds from start
stamps = np.array((data.index - data.index[0]).total_seconds())

# Extract position data as one big [N x 3] array
position = data[["x_trans", "y_trans", "z_trans"]].to_numpy()/1000

# Extract quaternion data as one big [N x 4] array
quat = data[["w", "x_rot", "y_rot", "z_rot"]].to_numpy()

#########################################################################
# Now you can construct a MocapTrajectory object from the data.
# This will construct an interpolating smoothing spline internally
mocap = MocapTrajectory(stamps, position, quat)

# Now you can sample at any time you want (in seconds from start)
query_stamps = np.arange(0, mocap.stamps[-1], 0.1)
pos = mocap.position(query_stamps)
quat = mocap.quaternion(query_stamps)
vel = mocap.velocity(query_stamps)
accel = mocap.acceleration(query_stamps)
imu_accel = mocap.accelerometer(query_stamps)
omega = mocap.angular_velocity(query_stamps)
is_static = mocap.is_static(query_stamps)

# Helper function to plot the data
import matplotlib.pyplot as plt
mocap.plot()
plt.show()