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
from matplotlib import pyplot as plt


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

    def plot(
        self, mocap: MocapTrajectory = None, axs: List[plt.Axes] = None
    ) -> Tuple[List[plt.Figure], List[plt.Axes]]:

        # Plot gyro
        if axs is None:
            fig1, gyro_axs = plt.subplots(3, 1, sharex=True)
            fig2, accel_axs = plt.subplots(3, 1, sharex=True)
            fig = [fig1, fig2]
            axs = np.concatenate([gyro_axs, accel_axs])
        else:
            fig = [axs[0].figure, axs[3].figure]
            gyro_axs = axs[:3]
            accel_axs = axs[3:]

        gyro_mocap = mocap.angular_velocity(self.stamps)
        gyro = self.angular_velocity
        gyro_axs[0].plot(self.stamps, gyro[:, 0], label="IMU")
        gyro_axs[1].plot(self.stamps, gyro[:, 1])
        gyro_axs[2].plot(self.stamps, gyro[:, 2])

        if mocap is not None:
            gyro_axs[0].plot(self.stamps, gyro_mocap[:, 0], label="Mocap")
            gyro_axs[1].plot(self.stamps, gyro_mocap[:, 1])
            gyro_axs[2].plot(self.stamps, gyro_mocap[:, 2])

        gyro_axs[0].legend()
        gyro_axs[-1].set_xlabel("Time (s)")
        gyro_axs[0].set_ylabel("X (rad/s)")
        gyro_axs[1].set_ylabel("Y (rad/s)")
        gyro_axs[2].set_ylabel("Z (rad/s)")
        gyro_axs[0].set_title("Rate gyro")

        # Plot accelerometer
        accel_mocap = mocap.accelerometer(self.stamps)
        accel = self.acceleration
        accel_axs[0].plot(self.stamps, accel[:, 0], label="IMU")
        accel_axs[1].plot(self.stamps, accel[:, 1])
        accel_axs[2].plot(self.stamps, accel[:, 2])

        if mocap is not None:
            accel_axs[0].plot(self.stamps, accel_mocap[:, 0], label="Mocap")
            accel_axs[1].plot(self.stamps, accel_mocap[:, 1])
            accel_axs[2].plot(self.stamps, accel_mocap[:, 2])

        accel_axs[0].legend()
        accel_axs[-1].set_xlabel("Time (s)")
        accel_axs[0].set_ylabel("X (m/s^2)")
        accel_axs[1].set_ylabel("Y (m/s^2)")
        accel_axs[2].set_ylabel("Z (m/s^2)")
        accel_axs[0].set_title("Accelerometer")

        return fig, axs

    def filter(
        self, lowcut_freq: float, highcut_freq: float = None, order: int = 4
    ):
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
        sos = signal.butter(
            order, cutoff_freq, fitlertype, fs=avg_freq, output="sos"
        )
        self.acceleration = signal.sosfiltfilt(sos, self.acceleration, axis=0)
        self.angular_velocity = signal.sosfiltfilt(
            sos, self.angular_velocity, axis=0
        )

    def apply_calibration(
        self,
        gyro_bias: np.ndarray = None,
        accel_bias: np.ndarray = None,
        gyro_scale: np.ndarray = None,
        accel_scale: np.ndarray = None,
    ) -> "IMUData":
        """
        Applies corrections to the IMU data and returns a new IMUData object

        Parameters
        ----------
        gyro_bias : np.ndarray, optional
            bias, by default None
        accel_bias : np.ndarray, optional
            bias, by default None
        gyro_scale : np.ndarray, optional
            scale factor, by default None. should be close to 1
        accel_scale : np.ndarray, optional
            scale factor, by default None. should be close to 1

        Returns
        -------
        IMUData
            corrected IMU data
        """

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

        return IMUData(
            self.stamps.copy(), acceleration.copy(), angular_velocity.copy()
        )

    def calibrate(
        self,
        mocap: MocapTrajectory,
        gyro_bias=True,
        accel_bias=True,
        body_frame=True,
        world_frame=True,
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
        # Calibrate gyro and body frame first
        C_mb, _, b_gyro, _ = dead_reckoning_calibration(
            self,
            mocap,
            gyro_bias=gyro_bias,
            accel_bias=False,
            body_frame=body_frame,
            world_frame=False,
            position_error_weight=None,
            bias_error_weight=None,
            attitude_error_weight=1,
        )

        # Apply the correction
        imu_gyro_calib = self.apply_calibration(gyro_bias=b_gyro)
        mocap_gyro_calib = mocap.rotate_body_frame(C_mb)

        # Then calibrate the accel bias and world frame
        _, C_wl, _, b_accel = direct_spline_calibration(
            imu_gyro_calib,
            mocap_gyro_calib,
            gyro_bias=False,
            accel_bias=accel_bias,
            body_frame=False,
            world_frame=world_frame,
        )

        grav_angles = np.degrees(SO3.Log(C_wl).ravel()[:2])
        g_a = C_wl @ np.array([0, 0, -9.80665])
        phi_mb = SO3.Log(C_mb).ravel()
        print(f"{mocap.frame_id} gyro bias                     : {b_gyro}")
        print(
            f"{mocap.frame_id} body frame error (degrees)    : {np.degrees(phi_mb)}"
        )
        print(f"{mocap.frame_id} accel bias                    : {b_accel}")
        print(f"{mocap.frame_id} gravity roll/pitch (degrees)  : {grav_angles}")
        print(f"{mocap.frame_id} gravity vector in mocap frame : {g_a}")
        return C_mb, C_wl, b_gyro, b_accel


def dead_reckoning_calibration(
    imu: IMUData,
    mocap: MocapTrajectory,
    gyro_bias=True,
    accel_bias=True,
    body_frame=True,
    world_frame=True,
    window_size=500,
    attitude_error_weight=1e6,
    position_error_weight=1,
    bias_error_weight=1e4,
):
    g_l = np.array([0, 0, -9.80665]).reshape((-1, 1))
    is_static = mocap.is_static(imu.stamps)

    ##### Initial guess for the bias using the static moments.
    static_accel = imu.acceleration[is_static, :]
    static_gyro = imu.angular_velocity[is_static, :]

    if gyro_bias:
        b_gyro = np.mean(static_gyro, axis=0)
    else:
        b_gyro = np.zeros(3)

    C_wm_static = mocap.rot_matrix(imu.stamps[is_static])
    C_mw_static = np.transpose(C_wm_static, [0, 2, 1])
    if accel_bias:
        b_accel = np.mean(C_mw_static @ g_l.ravel() + static_accel, axis=0)
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
    slices = [
        slice(i, i + window_size) for i in range(0, data_length, window_size)
    ]

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
    C_wm_i = mocap.rot_matrix(start_stamps)
    C_wm_j = mocap.rot_matrix(stop_stamps)
    C_mw_j = np.transpose(C_wm_j, [0, 2, 1])
    v_i = mocap.velocity(start_stamps)
    v_j = mocap.velocity(stop_stamps)
    start_is_static = mocap.is_static(start_stamps)
    stop_is_static = mocap.is_static(stop_stamps)
    v_i[start_is_static, :] = 0  # Spline is noisy. Zero out static velocities.
    v_j[stop_is_static, :] = 0
    DT = (stop_stamps - start_stamps).reshape((-1, 1))

    # no point in optimizing gyro bias. We already know very accurately.
    gyro_bias = False
    imu_chunks[:, :3, :] -= b_gyro.reshape((-1, 1))

    ##### Compute the error function for the least squares solver
    def rmi_error(x: np.ndarray):
        b_gyro, b_accel, C_mb, C_wl = _unpack_optimization_variables(
            x, gyro_bias, accel_bias, body_frame, world_frame
        )

        gyro_calib = C_mb @ (imu_chunks[:, :3, :] - b_gyro.reshape((-1, 1)))
        accel_calib = C_mb @ (imu_chunks[:, 3:, :] - b_accel.reshape((-1, 1)))
        g_w = (C_wl @ g_l).ravel()
        DR, DV, DC = compute_rmi_batch(chunk_stamps, gyro_calib, accel_calib)
        # v_j_est = v_i + g_w * DT + bmv(C_wm_i, DV)
        # e_v = (v_j - v_j_est)

        # Construct the error vector. Omit terms with zero, False, or None weight.
        e = []
        if position_error_weight:
            r_j_est = r_i + v_i * DT + 0.5 * g_w * DT**2 + bmv(C_wm_i, DR)
            e_r = r_j - r_j_est
            e.append(position_error_weight * e_r.ravel() / e_r.size)
        if attitude_error_weight:
            C_wm_j_est = C_wm_i @ DC
            e_C = blog_so3(C_mw_j @ C_wm_j_est)
            e.append(attitude_error_weight * e_C.ravel() / e_C.size)
        if bias_error_weight:
            e_bias = (
                C_mw_static @ g_w
                + (C_mb @ (static_accel - b_accel.ravel()).T).T
            )
            e.append(bias_error_weight * e_bias.ravel() / e_bias.size)

        return np.concatenate(e, axis=0)

    ##### Optimization
    print("Calibrating IMU...")
    x0 = []
    if gyro_bias:
        x0.append(b_gyro)
    if accel_bias:
        x0.append(b_accel)
    if body_frame:
        x0.append(np.array([0, 0, 0]))
    if world_frame:
        x0.append(np.array([0, 0, 0]))

    if len(x0) > 0:
        x0 = np.concatenate(x0, axis=0)

        result = least_squares(rmi_error, x0, verbose=2, jac="3-point")
        x_opt = result.x
    else:
        x_opt = []

    _, b_accel, C_mb, C_wl = _unpack_optimization_variables(
        x_opt, gyro_bias, accel_bias, body_frame, world_frame
    )
    return C_mb, C_wl, b_gyro, b_accel


def _unpack_optimization_variables(
    x, gyro_bias, accel_bias, body_frame, world_frame
):
    idx = 0
    if gyro_bias:
        b_gyro = x[idx : idx + 3]
        idx += 3
    else:
        b_gyro = np.zeros(3)

    if accel_bias:
        b_accel = x[idx : idx + 3]
        idx += 3
    else:
        b_accel = np.zeros(3)

    if body_frame:
        C_mb = SO3.Exp(x[idx : idx + 3])
        idx += 3
    else:
        C_mb = np.eye(3)

    if world_frame:
        C_wl = SO3.Exp([[x[idx]], [x[idx + 1]], [0]])
        idx += 2
    else:
        C_wl = np.eye(3)
    return b_gyro, b_accel, C_mb, C_wl


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


def direct_spline_calibration(
    imu: IMUData,
    mocap: MocapTrajectory,
    gyro_bias=True,
    accel_bias=True,
    body_frame=True,
    world_frame=True,
):
    g_l = np.array([0, 0, -9.80665]).reshape((-1, 1))
    is_static = mocap.is_static(imu.stamps)

    ##### Initial guess for the bias using the static moments.
    static_accel = imu.acceleration[is_static, :]
    static_gyro = imu.angular_velocity[is_static, :]

    if gyro_bias:
        b_gyro = np.mean(static_gyro, axis=0)
    else:
        b_gyro = np.zeros(3)

    C_wm_static = mocap.rot_matrix(imu.stamps[is_static])
    C_mw_static = np.transpose(C_wm_static, [0, 2, 1])
    if accel_bias:
        b_accel = np.mean(C_mw_static @ g_l.ravel() + static_accel, axis=0)
    else:
        b_accel = np.zeros(3)

    # no point in optimizing gyro bias. We already know very accurately.
    gyro_bias = False

    C_wm = mocap.rot_matrix(imu.stamps)
    C_mw = np.transpose(C_wm, [0, 2, 1])

    a_m = mocap.acceleration(imu.stamps)
    a_m[is_static] = 0  # We know these should be exactly 0
    w_m = mocap.angular_velocity(imu.stamps)
    w_m[is_static] = 0
    ##### Compute the error function for the least squares solver
    def imu_error(x: np.ndarray):
        b_gyro, b_accel, C_mb, C_wl = _unpack_optimization_variables(
            x, gyro_bias, accel_bias, body_frame, world_frame
        )

        g_m = (C_mw @ C_wl @ g_l).squeeze().T
        w_b = imu.angular_velocity - b_gyro
        a_b = imu.acceleration - b_accel

        error = np.concatenate(
            [
                (C_mb @ w_b.T) - w_m.T,
                (C_mb @ a_b.T + g_m) - a_m.T,
            ],
            axis=0,
        ).ravel()
        return error

    ##### Optimization
    print("Calibrating IMU...")
    x0 = []
    if gyro_bias:
        x0.append(b_gyro)
    if accel_bias:
        x0.append(b_accel)
    if body_frame:
        x0.append(np.array([0, 0, 0]))
    if world_frame:
        x0.append(np.array([0, 0, 0]))
    x0 = np.concatenate(x0, axis=0)

    result = least_squares(
        imu_error, x0, verbose=2, jac="2-point", loss="cauchy"
    )

    _, b_accel, C_mb, C_wl = _unpack_optimization_variables(
        result.x, gyro_bias, accel_bias, body_frame, world_frame
    )

    return C_mb, C_wl, b_gyro, b_accel
