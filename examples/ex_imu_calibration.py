from pymocap import MocapTrajectory, IMUData
import rosbag

filename = "data/imu_calib.bag"
agent = "ifo001"

# Extract data
with rosbag.Bag(filename, "r") as bag:
    imu = IMUData.from_bag(bag, f"/{agent}/mavros/imu/data_raw")
    mocap = MocapTrajectory.from_bag(bag, agent)

# Do calibration. See function documentation for interpretation of the output.
C_bm, C_wl, gyro_bias, accel_bias = imu.calibrate(mocap)

# Apply calibration results to the data. This modifies the internal data in place.
# TODO: should we have it return a new calibrated object instead?
# i.e. imu_calib = imu.apply_calibration(...)
#      mocap_calib = mocap.rotate_body_frame(...)
imu.apply_calibration(gyro_bias = gyro_bias, accel_bias = accel_bias)
mocap.rotate_body_frame(C_bm)
mocap.rotate_world_frame(C_wl)

# At this point, the data inside (mocap, imu) is calibrated and ready for use.