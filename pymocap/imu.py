from typing import Dict, List, Tuple, Any
import numpy as np
from sensor_msgs.msg import Imu
from scipy.optimize import least_squares
from scipy import signal
from .mocap import MocapTrajectory
from pylie import SO3
from .utils import bmv, bag_to_list, blog_so3, bexp_so3
from pynav.lib.states import SE23State
from pynav.lib.imu import IMU, IMUKinematics


class IMUData:
    def __init__(
        self,
        stamps: np.ndarray,
        acceleration: np.ndarray,
        angular_velocity: np.ndarray,
    ):
        self.stamps = stamps
        self.acceleration = acceleration
        self.angular_velocity = angular_velocity

    @staticmethod
    def from_ros(imu_data: List[Imu]):

        stamps = []
        acceleration = []
        angular_velocity = []

        for p in imu_data:
            stamps.append(p.header.stamp.to_sec())
            acceleration.append(
                [
                    p.linear_acceleration.x,
                    p.linear_acceleration.y,
                    p.linear_acceleration.z,
                ]
            )
            angular_velocity.append(
                [
                    p.angular_velocity.x,
                    p.angular_velocity.y,
                    p.angular_velocity.z,
                ]
            )

        stamps = np.array(stamps)
        acceleration = np.array(acceleration)
        angular_velocity = np.array(angular_velocity)
        return IMUData(stamps, acceleration, angular_velocity)

    @staticmethod
    def from_bag(bag, topic: str):
        imu_data = bag_to_list(bag, topic)
        return IMUData.from_ros(imu_data)

    def to_pynav(self, state_id=None) -> List[IMU]:
        acc = self.acceleration.copy()
        gyro = self.angular_velocity.copy()

        data = []
        for i in range(len(self.stamps)):
            u = IMU(gyro[i, :], acc[i, :], self.stamps[i], state_id=state_id)
            data.append(u)
        return data

    def filter(self, lowcut_freq: float, highcut_freq: float = None, order: int = 4):
        """
        Filters the IMU data using a Butterworth filter.

        Parameters
        ----------
        cutoff_freq : float
            The lower cutoff frequency of the filter in Hz.
        highcut_freq : float, optional
            The higher cutoff frequency of the filter in Hz, by default None.
            If this is provided, the filter becomes a "band stop" filter that
            filters out the frequencies in between lowcut_freq and highcut_freq.
            Otherwise, the filter is a low-pass attenuating all frequencies
            above ``lowcut_freq``
        order : int, optional
            The order of the filter, by default 4
        """
        if highcut_freq is None:
            cutoff_freq = lowcut_freq
            fitlertype = "lowpass"
        else:
            cutoff_freq = [lowcut_freq, highcut_freq]
            fitlertype = "bandstop"
        avg_freq = 1 / ((self.stamps[-1] - self.stamps[0]) / self.stamps.size)
        sos = signal.butter(order, cutoff_freq, fitlertype, fs=avg_freq, output="sos")
        self.acceleration = signal.sosfiltfilt(sos, self.acceleration, axis=0)
        self.angular_velocity = signal.sosfiltfilt(sos, self.angular_velocity, axis=0)

    def apply_calibration(
        self,
        gyro_bias: np.ndarray = None,
        accel_bias: np.ndarray = None,
        gyro_scale: np.ndarray = None,
        accel_scale: np.ndarray = None,
    ) -> "IMUData":

        angular_velocity: np.ndarray = self.angular_velocity
        acceleration: np.ndarray = self.acceleration
        if gyro_bias is not None:
            gyro_bias = np.array(gyro_bias).ravel()
            angular_velocity = angular_velocity - gyro_bias
        if gyro_scale is not None:
            gyro_scale = np.array(gyro_scale).ravel()
            angular_velocity = angular_velocity * gyro_scale
        if accel_bias is not None:
            accel_bias = np.array(accel_bias).ravel()
            acceleration = acceleration - accel_bias
        if accel_scale is not None:
            accel_scale = np.array(accel_scale).ravel()
            acceleration = acceleration * accel_scale

        return IMUData(self.stamps.copy(), acceleration.copy(), angular_velocity.copy())

    def calibrate(
        self,
        mocap: MocapTrajectory,
        body_frame=True,
        world_frame=True,
        gyro_bias=True,
        accel_bias=True,
    ):
        """
        Performs frame and bias calibration on the IMU data. Specifically,
        given mocap data, this function will estimate misalignments between
        1) the IMU frame and the mocap body frame, and 2) the mocap world frame
        and a "levelled" world frame aligned with gravity. In addition,
        gyro and accelerometer biases are estimated.

        Parameters
        ----------
        mocap : MocapTrajectory
            the mocap data to use for calibration
        do_gravity : bool, optional
            whether to estimate the mocap world frame misalignment, by default True

        Returns
        -------
        np.ndarray
            Mocap body frame to IMU frame rotation matrix
        np.ndarray
            Levelled world frame to mocap world frame rotation matrix
        np.ndarray
            gyro bias
        np.ndarray
            accel bias
        """

        # ########################################################
        # Gyro bias and body frame calibration

        # We first start by doing only gyro calibration to make the problem
        # simpler. We can then use these calibrated results to an
        # accelerometer calibration procedure.
        C_bm, gyro_bias = _preint_gyro_calibration(self, mocap)

        mocap_gyro_calib = mocap.rotate_body_frame(C_bm)
        imu_gyro_calib = self.apply_calibration(gyro_bias=gyro_bias)

        # ########################################################
        # Accelerometer bias and world frame calibration
        C_wl, accel_bias = _preint_accel_calibration(imu_gyro_calib, mocap_gyro_calib)
        return C_bm, C_wl, gyro_bias, accel_bias


def _preint_gyro_calibration(
    imu: IMUData, mocap: MocapTrajectory, gyro_bias=True, body_frame=True
):
    is_static = mocap.is_static(imu.stamps)
    static_gyro = imu.angular_velocity[is_static, :]

    if gyro_bias:
        b_gyro = np.mean(static_gyro, axis=0)
    else:
        b_gyro = np.zeros(3)

    window_size = 500  # L

    # Get only the dynamic portion of the trajectory
    dynamic_start = np.nonzero(~is_static)[0][0]
    dynamic_end = np.nonzero(~is_static)[0][-1]
    dynamic_range = slice(dynamic_start, dynamic_end)
    gyro = (imu.angular_velocity - b_gyro).T
    stamps = imu.stamps

    data_length = gyro.shape[1]

    # Create a slice object for each chunk
    slices = [slice(i, i + window_size) for i in range(0, data_length, window_size)]

    # If last slice is not a full chunk, remove it.
    if slices[-1].stop > data_length:
        slices = slices[:-1]

    # [num_chunks x window_size]
    chunk_stamps = np.array([stamps[s] for s in slices])

    start_stamps = chunk_stamps[:, 0]
    stop_stamps = chunk_stamps[:, -1]

    # Seperate gyro data into chunks of size [num_chunks x 3 x window_size]
    gyro_chunks = np.array([gyro[:, s] for s in slices])

    # Ground truth DCMs [num_chunks x 3 x 3]
    C_ab_i = mocap.rot_matrix(start_stamps)
    C_ab_j = mocap.rot_matrix(stop_stamps)

    def rmi_error(x: np.ndarray):
        C_mb = SO3.Exp(x[:3])  # Gyro body frame to mocap frame DCM
        calibrated_chunks = C_mb @ gyro_chunks
        DC = compute_attitude_rmi_batch(chunk_stamps, calibrated_chunks)
        C_ab_j_est = C_ab_i @ DC

        e_C = blog_so3(np.transpose(C_ab_j, [0, 2, 1]) @ C_ab_j_est)
        return e_C.ravel()

    print("Calibrating gyro frame...")
    if body_frame:
        result = least_squares(
            rmi_error,
            np.zeros((3,)),
            verbose=2,
            max_nfev=50,
        )
        phi_mb = result.x
        C_bm = SO3.Exp(phi_mb).T
    else:
        C_bm = np.eye(3)
        phi_mb = np.zeros(3)

    print(f"{mocap.frame_id} gyro bias                  : {b_gyro}")
    print(f"{mocap.frame_id} gyro frame error (degrees) : {np.degrees(-phi_mb)}")
    return C_bm, b_gyro


def _preint_accel_calibration(
    imu: IMUData, mocap: MocapTrajectory, accel_bias=True, world_frame=True
):
    window_size = 1000  # L
    g_l = np.array([0, 0, -9.80665]).reshape((-1, 1))
    is_static = mocap.is_static(imu.stamps)

    ##### Initial guess for the bias using the static moments.
    static_accel = imu.acceleration[is_static, :]
    C_wb_static = mocap.rot_matrix(imu.stamps[is_static])
    C_bw_static = np.transpose(C_wb_static, [0, 2, 1])
    if accel_bias:
        b_accel = np.mean(C_bw_static @ g_l.ravel() + static_accel, axis=0)
    else:
        b_accel = np.zeros(3)

    ##### Get only the dynamic portion of the trajectory
    dynamic_start = np.nonzero(~is_static)[0][0]
    dynamic_end = np.nonzero(~is_static)[0][-1]
    dynamic_range = slice(dynamic_start, dynamic_end)
    accel = imu.acceleration[dynamic_range].T
    gyro = imu.angular_velocity[dynamic_range].T
    stamps = imu.stamps[dynamic_range]

    ##### Divide dynamic portion into chunks (one `slice` per chunk)
    data_length = accel.shape[1]
    slices = [slice(i, i + window_size) for i in range(0, data_length, window_size)]

    if slices[-1].stop > data_length:
        # If last slice is not a full chunk, remove it.
        slices = slices[:-1]

    # [num_chunks x window_size]
    chunk_stamps = np.array([stamps[s] for s in slices])
    start_stamps = chunk_stamps[:, 0]
    stop_stamps = chunk_stamps[:, -1]

    # 6 x data_size
    imu_data = np.vstack([gyro, accel])
    # N x 6 x window_size
    imu_chunks = np.array([imu_data[:, s] for s in slices])

    ##### Get ground truth value at beginning and end of each chunk
    start_stamps = chunk_stamps[:, 0]
    stop_stamps = chunk_stamps[:, -1]

    r_i = mocap.position(start_stamps)
    r_j = mocap.position(stop_stamps)
    C_wb_i = mocap.rot_matrix(start_stamps)
    v_i = mocap.velocity(start_stamps)
    v_j = mocap.velocity(stop_stamps)
    start_is_static = mocap.is_static(start_stamps)
    stop_is_static = mocap.is_static(stop_stamps)
    v_i[start_is_static, :] = 0  # Spline is noisy. Zero out static velocities.
    v_j[stop_is_static, :] = 0
    DT = (stop_stamps - start_stamps).reshape((-1, 1))

    ##### Compute the error function for the least squares solver
    def rmi_error(x: np.ndarray):
        if world_frame:
            C_wl = SO3.Exp([x[0], x[1], 0])  # Levelled frame to mocap world frame
        else:
            C_wl = np.eye(3)

        if accel_bias:
            bias = x[2:].reshape((1, 3, 1))
        else:
            bias = np.zeros((1, 3, 1))

        gyro_calib = imu_chunks[:, :3, :]
        accel_calib = imu_chunks[:, 3:, :] - bias
        DR, DV, DC = compute_rmi_batch(chunk_stamps, gyro_calib, accel_calib)
        g_w = (C_wl @ g_l).ravel()
        r_j_est = r_i + v_i * DT + 0.5 * g_w * DT**2 + bmv(C_wb_i, DR)
        e_r = r_j - r_j_est
        e_r = e_r / e_r.size
        # v_j_est = v_i + g_w * DT + bmv(C_wb_i, DV)
        # e_v = (v_j - v_j_est)
        # #e_v = e_v[np.logical_and(start_is_static, stop_is_static),:].ravel()

        e_bias = C_bw_static @ g_w + (static_accel - bias.ravel())
        # e_bias = e_bias[:4000,:]
        e_bias = e_bias / e_bias.size

        return np.concatenate([e_r.ravel(), 1e6 * e_bias.ravel()], axis=0)

    ##### Optimization
    print("Calibrating accel...")
    x0 = np.zeros((5,))
    x0[2:] = b_accel.ravel()
    result = least_squares(rmi_error, x0, verbose=2, loss="soft_l1")
    x = result.x
    C_wl = SO3.Exp([x[0], x[1], 0])
    b_accel = x[2:]
    grav_angles = np.degrees(SO3.Log(C_wl).ravel()[:2])
    g_a = C_wl @ np.array([0, 0, -9.80665])
    print(f"{mocap.frame_id} accel bias                    : {b_accel}")
    print(f"{mocap.frame_id} gravity roll/pitch (degrees)  : {grav_angles}")
    print(f"{mocap.frame_id} gravity vector in mocap frame : {g_a}")

    return C_wl, b_accel


def compute_attitude_rmi_batch(stamps: np.ndarray, gyro: np.ndarray):
    """
    Given a batch of IMU measurement chunks, it will compute the RMIs
    associated with each sequence in the batch.

    Parameters
    ----------
    stamps : np.ndarray with shape (N, len)
        Timestamps of the IMU measurements.
    gyro : np.ndarray with shape (N, 3, len)
        Rate gyro measurements
    accel : np.ndarray with shape (N, 3, len)
        accelerometer measurements

    Returns
    -------
    Tuple[np.ndarray(N,3), np.ndarray(N,3), np.ndarray(N,3,3)]
        Position, velocity, attitude RMIs
    """
    dim_batch = gyro.shape[0]
    N = gyro.shape[2]
    t = stamps
    DC = np.tile(np.eye(3), (dim_batch, 1, 1))
    for idx in range(1, N):
        dt = (t[:, idx] - t[:, idx - 1]).reshape((-1, 1))
        w = gyro[:, :, idx - 1]

        Om = bexp_so3(w * dt)

        DC = DC @ Om

    return DC


def compute_rmi_batch(stamps: np.ndarray, gyro: np.ndarray, accel: np.ndarray):
    """
    Given a batch of IMU measurement chunks, it will compute the RMIs
    associated with each sequence in the batch.

    Parameters
    ----------
    stamps : np.ndarray with shape (N, len)
        Timestamps of the IMU measurements.
    gyro : np.ndarray with shape (N, 3, len)
        Rate gyro measurements
    accel : np.ndarray with shape (N, 3, len)
        accelerometer measurements

    Returns
    -------
    Tuple[np.ndarray(N,3), np.ndarray(N,3), np.ndarray(N,3,3)]
        Position, velocity, attitude RMIs
    """
    dim_batch = gyro.shape[0]
    N = gyro.shape[2]
    t = stamps
    DC = np.tile(np.eye(3), (dim_batch, 1, 1))
    DV = np.zeros((dim_batch, 3))
    DR = np.zeros((dim_batch, 3))
    for idx in range(1, N):
        dt = (t[:, idx] - t[:, idx - 1])[:, None]
        w = gyro[:, :, idx - 1]
        a = accel[:, :, idx - 1]
        DR = DR + DV * dt + 0.5 * bmv(DC, (a * (dt**2)))
        DV = DV + bmv(DC, (a * dt))

        Om = bexp_so3(w * dt)

        DC = DC @ Om

    return DR, DV, DC
