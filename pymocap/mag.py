from typing import Dict, List, Tuple, Any
import numpy as np
from pynav.lib.models import Magnetometer
from .utils import bag_to_list, bmv
from .mocap import MocapTrajectory
from matplotlib import pyplot as plt
from sensor_msgs.msg import MagneticField
from scipy.optimize import least_squares
from pylie import SO3
import scipy
from scipy.linalg import block_diag
from scipy.linalg import sqrtm


class MagnetometerData:
    def __init__(
        self,
        stamps: np.ndarray,
        magnetic_field: np.ndarray,
    ):
        self.stamps = stamps
        self.magnetic_field = magnetic_field

    @staticmethod
    def from_ros(imu_data: List[MagneticField]):

        stamps = []
        magnetic_field = []

        for p in imu_data:
            stamps.append(p.header.stamp.to_sec())
            magnetic_field.append(
                [
                    p.magnetic_field.x,
                    p.magnetic_field.y,
                    p.magnetic_field.z,
                ]
            )
        stamps = np.array(stamps)
        magnetic_field = np.array(magnetic_field)
        return MagnetometerData(stamps, magnetic_field)

    @staticmethod
    def from_bag(bag, topic: str):
        mag_data = bag_to_list(bag, topic)
        return MagnetometerData.from_ros(mag_data)

    def to_pynav(self, state_id=None) -> List[Magnetometer]:
        pass

    def plot(
        self,
        mocap: MocapTrajectory,
        world_frame=False,
        mag_vector: List[float] = None,
        axs=None,
    ):
        # Plot gyro
        if axs is None:
            fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
            lim = 2*np.mean(np.linalg.norm(self.magnetic_field, axis=1))
            axs[0].set_ylim(-lim, lim)

        else:
            fig = axs[0].figure()

        C_ab = mocap.rot_matrix(self.stamps)
        mag = self.magnetic_field

        if world_frame:
            mag = bmv(C_ab, mag)

        axs[0].plot(self.stamps, mag[:, 0], label="Magnetometer")
        axs[1].plot(self.stamps, mag[:, 1])
        axs[2].plot(self.stamps, mag[:, 2])

        if mag_vector is not None:
            if world_frame:
                axs[0].plot(self.stamps, mag_vector[0])
                axs[1].plot(self.stamps, mag_vector[1])
                axs[2].plot(self.stamps, mag_vector[2])
            else:
                C_ba = np.transpose(C_ab, [0,2,1])
                mag_vector = C_ba @ mag_vector
                axs[0].plot(self.stamps, mag_vector[:, 0])
                axs[1].plot(self.stamps, mag_vector[:, 1])
                axs[2].plot(self.stamps, mag_vector[:, 2])

    def calibrate(self, mocap: MocapTrajectory = None):
        ##### Get only the dynamic portion of the trajectory
        is_static = mocap.is_static(self.stamps)
        dynamic_start = np.nonzero(~is_static)[0][0]
        dynamic_end = np.nonzero(~is_static)[0][-1]
        dynamic_range = slice(dynamic_start, dynamic_end)
        mag_data = self.magnetic_field[dynamic_range].T
        stamps = self.stamps[dynamic_range]
        scaling_factor = np.mean(np.linalg.norm(mag_data, axis=0))
        mag_data = mag_data/scaling_factor
        M, n, d = self._ellipsoid_fit(mag_data)

        M_inv = np.linalg.inv(M)
        bias = -np.dot(M_inv, n)
        A_inv = np.real(
            1 / np.sqrt(np.dot(n.T, np.dot(M_inv, n)) - d) * sqrtm(M)
        )


        # Get mag-to-body frame initial guess and dip angle.
        def mag_frame_error(x):
            d = x[0]
            R = SO3.Exp(x[1:4])
            e = d - np.sum(mag_data * ((R.T @ A_inv) @ (mag_data - bias)), axis=0)
            return e.ravel()

        result = least_squares(mag_frame_error, [1,0,0,0], verbose=2, loss="cauchy")
        x = result.x
        d = x[0]
        R = SO3.Exp(x[1:4])
        m_n = np.array([np.sqrt(1 - d**2), 0, -d])
        m_b = (R.T @ A_inv) @ (mag_data - bias)
        print(R)
        print(m_n)

        # Get world to magnetic north initial guess
        C_ab = mocap.rot_matrix(stamps)
        m_a_data = bmv(C_ab, m_b.T)
        def world_frame_error(x):
            C_an = SO3.Exp([0,0,x[0]])
            m_a = C_an @ m_n
            e = m_a - m_a_data
            return e.ravel()

        result = least_squares(world_frame_error, np.zeros(1), verbose=2, loss="cauchy")
        x = result.x

        C_an = SO3.Exp([0,0,x[0]])
        print(C_an)
        print(C_an @ m_n)


        # Final calibration 
        m_a_0 = C_an @ m_n
        D_0 = np.linalg.inv(R.T @ A_inv)
        bias_0 = bias 
        C_ba = np.transpose(mocap.rot_matrix(stamps), [0,2,1])
        def full_error(x):
            D = x[0:9].reshape((3,3))
            b = x[9:12].reshape((-1,1))
            phi = x[12:14]
            m_a =  SO3.Exp([0, phi[0], phi[1]]) @ m_a_0
            y_hat = D @ (C_ba @ m_a).T + b
            e = mag_data - y_hat
            return e.ravel() 

        x_0 = np.hstack([D_0.ravel(), bias_0.ravel(), [0,0]])
        results =least_squares(full_error, x_0, loss="cauchy")
        x = results.x

        D = x[0:9].reshape((3,3))
        b = x[9:12].reshape((-1,1))
        phi = x[12:14]
        m_a =  SO3.Exp([0, phi[0], phi[1]]) @ m_a_0

        print(D)
        print(b)
        print(m_a)


        D_inv = np.linalg.inv(D)
        return D*scaling_factor, b*scaling_factor, m_a 

    def apply_calibration(self, distortion, bias): 
        D_inv = np.linalg.inv(distortion)
        corrected_mag = D_inv @ (self.magnetic_field.T - bias)
        return MagnetometerData(self.stamps, corrected_mag.T)

    def _ellipsoid_fit(self, s):
        """
        Estimate ellipsoid parameters from a set of points.

        Parameters
        ----------
        s : array_like
            The samples (M,N) where M=3 (x,y,z) and N=number of samples.

        Returns
        -------
        M, n, d : array_like, array_like, float
            The ellipsoid parameters M, n, d.

        References
        ----------
        .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
            fitting," in Geometric Modeling and Processing, 2004.
            Proceedings, vol., no., pp.335-340, 2004
        """

        # D (samples)
        D = np.array(
            [
                s[0] ** 2.0,
                s[1] ** 2.0,
                s[2] ** 2.0,
                2.0 * s[1] * s[2],
                2.0 * s[0] * s[2],
                2.0 * s[0] * s[1],
                2.0 * s[0],
                2.0 * s[1],
                2.0 * s[2],
                np.ones_like(s[0]),
            ]
        )

        # S, S_11, S_12, S_21, S_22 (eq. 11)
        S = np.dot(D, D.T)
        S_11 = S[:6, :6]
        S_12 = S[:6, 6:]
        S_21 = S[6:, :6]
        S_22 = S[6:, 6:]

        # C (Eq. 8, k=4)
        C = np.array(
            [
                [-1, 1, 1, 0, 0, 0],
                [1, -1, 1, 0, 0, 0],
                [1, 1, -1, 0, 0, 0],
                [0, 0, 0, -4, 0, 0],
                [0, 0, 0, 0, -4, 0],
                [0, 0, 0, 0, 0, -4],
            ]
        )

        # v_1 (eq. 15, solution)
        E = np.dot(
            np.linalg.inv(C),
            S_11 - np.dot(S_12, np.dot(np.linalg.inv(S_22), S_21)),
        )

        E_w, E_v = np.linalg.eig(E)

        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0:
            v_1 = -v_1

        # v_2 (eq. 13, solution)
        v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

        # quadric-form parameters
        M = np.array(
            [
                [v_1[0], v_1[3], v_1[4]],
                [v_1[3], v_1[1], v_1[5]],
                [v_1[4], v_1[5], v_1[2]],
            ]
        )
        n = np.array([[v_2[0]], [v_2[1]], [v_2[2]]])
        d = v_2[3]

        return M, n, d
