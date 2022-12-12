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
        # I know this function looks dumb right now but before i was doing 
        # other things. We can keep this if we want to modify the 
        # calibration procedure such that we first do gyro only by setting 
        # some toggles, and then do accel only, and then do both (or something).
        C_bm, C_wl, b_gyro, b_accel = _preint_calibration(
            self, mocap, gyro_bias, accel_bias, body_frame, world_frame
        )
        return C_bm, C_wl, b_gyro, b_accel




def _preint_calibration(
    imu: IMUData,
    mocap: MocapTrajectory,
    gyro_bias=True,
    accel_bias=True,
    body_frame=True,
    world_frame=True,
    window_size=1000,
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

        gyro_calib = C_mb @ (imu_chunks[:, :3, :] - b_gyro.reshape((-1,1)))
        accel_calib = C_mb @ (imu_chunks[:, 3:, :] - b_accel.reshape((-1,1)))
        DR, DV, DC = compute_rmi_batch(chunk_stamps, gyro_calib, accel_calib)
        g_w = (C_wl @ g_l).ravel()
        r_j_est = r_i + v_i * DT + 0.5 * g_w * DT**2 + bmv(C_wm_i, DR)
        C_wm_j_est = C_wm_i @ DC
        e_r = r_j - r_j_est

        e_C = blog_so3(C_mw_j @ C_wm_j_est)
        # v_j_est = v_i + g_w * DT + bmv(C_wb_i, DV)
        # e_v = (v_j - v_j_est)
        # #e_v = e_v[np.logical_and(start_is_static, stop_is_static),:].ravel()

        e_bias = C_mw_static @ g_w + (C_mb @ (static_accel - b_accel.ravel()).T).T
        # e_bias = e_bias[:4000,:]
        return np.concatenate(
            [
                1e2*e_r.ravel() / e_r.size,
                1e8*e_C.ravel() / e_C.size,
                1e6*e_bias.ravel() / e_bias.size
,
            ],
            axis=0,
        )

    ##### Optimization

    print("Calibrating IMU...")
    x0 = []
    if gyro_bias:
        x0.append(b_gyro)
    if accel_bias:
        x0.append(b_accel)
    if body_frame:
        x0.append(np.array([0,0,0]))
    if world_frame:
        x0.append(np.array([0,0,0]))
    x0 = np.concatenate(x0, axis=0)

    result = least_squares(rmi_error, x0, verbose=2, loss="soft_l1")
    _, b_accel, C_mb, C_wl = _unpack_optimization_variables(
            result.x, gyro_bias, accel_bias, body_frame, world_frame
        )
    grav_angles = np.degrees(SO3.Log(C_wl).ravel()[:2])
    g_a = C_wl @ np.array([0, 0, -9.80665])
    C_bm = C_mb.T
    phi_bm = SO3.Log(C_bm).ravel()
    print(f"{mocap.frame_id} gyro bias                     : {b_gyro}")
    print(
        f"{mocap.frame_id} body frame error (degrees)    : {np.degrees(-phi_bm)}"
    )
    print(f"{mocap.frame_id} accel bias                    : {b_accel}")
    print(f"{mocap.frame_id} gravity roll/pitch (degrees)  : {grav_angles}")
    print(f"{mocap.frame_id} gravity vector in mocap frame : {g_a}")

    return C_bm, C_wl, b_gyro, b_accel


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
