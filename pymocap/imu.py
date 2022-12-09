from typing import Dict, List, Tuple, Any
import numpy as np
from sensor_msgs.msg import  Imu
from pynav.lib.imu import IMU
from scipy.optimize import least_squares
from scipy import signal
from .mocap import MocapTrajectory
from pylie import SO3
from .utils import bmv


class IMUData:
    def __init__(self, stamps: np.ndarray, acceleration: np.ndarray, angular_velocity: np.ndarray):
        self.stamps =stamps
        self.acceleration = acceleration
        self.angular_velocity = angular_velocity
        self.raw_accel = self.acceleration
        self.raw_gyro = self.angular_velocity

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

    def to_pynav(self, state_id=None) -> List[IMU]:
        acc = self.acceleration.copy()
        gyro = self.angular_velocity.copy()

        data = []
        for i in range(len(self.stamps)):
            u = IMU(gyro[i, :], acc[i, :], self.stamps[i], state_id=state_id)
            data.append(u)
        return data

    def filter(self, lowcut_freq: float, highcut_freq=None, order: int = 4):
        """
        Filters the IMU data using a Butterworth filter.

        Parameters
        ----------
        cutoff_freq : float
            The cutoff frequency of the filter.
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
        gyro_scale: np.ndarray = None,
        accel_bias: np.ndarray = None,
        accel_scale: np.ndarray = None,
    ):
        if gyro_bias is not None:
            gyro_bias = np.array(gyro_bias).ravel()
            self.angular_velocity -= gyro_bias
        if gyro_scale is not None:
            gyro_scale = np.array(gyro_scale).ravel()
            self.angular_velocity *= gyro_scale
        if accel_bias is not None:
            accel_bias = np.array(accel_bias).ravel()
            self.acceleration -= accel_bias
        if accel_scale is not None:
            accel_scale = np.array(accel_scale).ravel()
            self.acceleration *= accel_scale

    def calibrate(self, mocap: MocapTrajectory, do_gravity = True):

        # ########################################################
        # Gyro bias and body frame calibration
        is_static = mocap.is_static(self.stamps)
        static_gyro = self.angular_velocity[is_static, :]
        gyro_bias = np.mean(static_gyro, axis=0)
        gyro_imu = self.angular_velocity - gyro_bias
        # gyro_mocap = mocap.angular_velocity(self.stamps)

        # print("Calibrating gyro frame...")
        # def gyro_frame_calib(phi_bm: np.ndarray):
        #     phi_bm = phi_bm.ravel()
        #     C_bm = SO3.Exp(phi_bm)
        #     e = C_bm @ gyro_mocap.T - gyro_imu.T
        #     return e.ravel()

        # result = least_squares(
        #     gyro_frame_calib,
        #     np.zeros((3,)),
        #     method="trf",
        #     verbose=2,
        #     loss="cauchy",
        # )
        # C_bm = SO3.Exp(result.x)
        C_bm = self._preint_att_calibration(mocap)

        print(f"{mocap.frame_id} gyro bias                  : {gyro_bias}")
        print(f"{mocap.frame_id} gyro frame error (degrees) : {np.degrees(SO3.Log(C_bm).ravel())}")

        # ########################################################
        # Accelerometer bias and world frame calibration
        accel_mocap = mocap.acceleration(self.stamps)
        C_wm = mocap.rot_matrix(self.stamps)
        C_mw = np.transpose(C_wm, [0, 2, 1])
        g_l = np.array([0, 0, -9.80665])

        def accel_frame_calib(x: np.ndarray):
            b = x[:3].ravel()

            if do_gravity:
                phi_aa = np.array([x[3], x[4], 0])
                C_wl = SO3.Exp(phi_aa)
            else:
                C_wl = np.identity(3)

            C_bw = C_bm @ C_mw
            accelerometer_mocap = bmv(
                C_bw, accel_mocap - (C_wl @ g_l.reshape((-1, 1))).ravel()
            )
            e = accelerometer_mocap - (self.acceleration - b)
            return e.ravel()

        print("Calibrating accelerometer frame...")
        result = least_squares(
            accel_frame_calib,
            np.zeros((5,)),
            method="trf",
            verbose=2,
            loss="cauchy",
        )

        #C_am = SO3.Exp(result.x[:3])
        accel_bias = result.x[:3]
        C_wl = SO3.Exp(np.array([result.x[3], result.x[4], 0]))
        g_a = (C_wl @ g_l.reshape((-1, 1))).ravel()
        # print(
        #     f"{mocap.frame_id} accel frame error (degrees)   : {np.degrees(result.x[:3])}"
        # )
        print(f"{mocap.frame_id} accel bias                    : {accel_bias}")
        print(
            f"{mocap.frame_id} gravity roll/pitch (degrees)  : {np.degrees(result.x[3:])}"
        )
        print(f"{mocap.frame_id} gravity vector in mocap frame : {g_a}")
        # ########################################################
        # ########################################################

        return C_bm, C_wl, gyro_bias, accel_bias

    def _preint_att_calibration(self, mocap: MocapTrajectory):
        window_size = 500  # L

        gyro = self.angular_velocity.T
        accel = self.acceleration.T  # replace here for only spots with mocap

        L = gyro.shape[1]
        start = np.array(range(0, L, window_size))
        stop = np.array(range(window_size, L + window_size, window_size))

        if stop[-1] > L:  # Omit last chunk if it doesnt fit
            start = start[:-1]
            stop = stop[:-1]

        slices = [
            slice(s, p) for s, p in zip(start, stop)
        ]  # can probably construct this better
        stamps_batch = np.array(
            [self.stamps[s] for s in slices]
        )  # N x window_size
        imu_data = np.vstack([gyro, accel])  # 6 x amount of meas
        imu_batch = np.array(
            [imu_data[:, s] for s in slices]
        )  # N x 6 x window_size
        start_stamps = stamps_batch[:, 0]
        stop_stamps = stamps_batch[:, -1]

        C_ab_i = mocap.rot_matrix(start_stamps)
        C_ab_j = mocap.rot_matrix(stop_stamps)

        def rmi_error(x: np.ndarray):
            dim_batch = imu_batch.shape[0]
            C_mb = SO3.Exp(x[:3])
            gyro_calib = C_mb @ imu_batch[:, :3, :]
            DC = compute_attitude_rmi_batch(stamps_batch, gyro_calib)
            C_ab_j_est = C_ab_i @ DC 

            e_C = np.array(
                [
                    SO3.Log(C_ab_j[i, :, :].T @ C_ab_j_est[i, :, :])
                    for i in range(dim_batch)
                ]
            )
            return e_C.ravel()

        print("Calibrating gyro frame...")
        result = least_squares(rmi_error, np.zeros((3,)), verbose=2, max_nfev=50, method="lm")
        phi_mb = result.x
        C_bm = SO3.Exp(phi_mb).T
        return C_bm

def compute_attitude_rmi_batch(stamps: np.ndarray, gyro: np.ndarray):
    """
    Given a batch of IMU measurement sequences, it will compute the RMIs
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
        dt = (t[:, idx] - t[:, idx - 1])
        w = gyro[:, :, idx - 1]

        # TODO. vectorizable in theory.
        Om = np.array([SO3.Exp(w[i, :] * dt[i]) for i in range(dim_batch)])

        DC = DC @ Om

    return DC