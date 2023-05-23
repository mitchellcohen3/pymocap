from typing import List, Any
import numpy as np
from csaps import csaps
from scipy.interpolate import interp1d
from .utils import bag_to_list, bquat_to_so3, bso3_to_quat
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from pylie import SO3

try:
    # pynav will be an optional dependency. If it is not installed, some
    # functions will not work. but thats okay.
    from pynav.lib.states import SE3State, SE23State

except ImportError:
    # If not installed, then create dummy classes so that the user can still
    # import this module.
    class SE3State:
        def __init__(self, *args, **kwargs):
            raise ImportError("pynav not installed.")
    
    class SE23State:
        def __init__(self, *args, **kwargs):
            raise ImportError("pynav not installed.")
     


class MocapTrajectory:
    """
    This class holds a mocap dataset and provides several convient getters.
    A smoothing spline is fit through the position and attitude data, giving access
    via its derivatives to estimated velocity and acceleration data. Furthermore,
    the spline can be queried at any point so that ground truth becomes available
    at any time.
    """

    def __init__(
        self,
        stamps: np.ndarray, 
        position_data: np.ndarray,
        quaternion_data: np.ndarray,
        frame_id: Any,
        quat_stamps: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        stamps : np.ndarray with shape (N,)
            Timestamps of the data
        position_data : np.ndarray with shape (N, 3)
            Position data where each row is a 3D position
        quaternion_data : np.ndarray with shape (N, 4)
            Attitude data where each row is a quaternion
        frame_id : Any
            Optional frame ID to assign to this data. Will be used as the state
            ID when converting to ``pynav`` states.
        """

        self.stamps = stamps
        self.raw_position = position_data
        self.raw_quaternion = quaternion_data
        self.frame_id = frame_id

        if quat_stamps is None:
            quat_stamps = stamps 

        self._fit_position_spline(self.stamps, self.raw_position)
        self._fit_quaternion_spline(quat_stamps, self.raw_quaternion)

        #:np.ndarray: Boolean array containing a static flag for each data point
        self.static_mask = self.get_static_mask(1, 0.0008)

    def _fit_position_spline(self, stamps, pos):
        # Fit splines
        self._pos_spline = csaps(stamps, pos.T, smooth=0.9999)

    def _fit_quaternion_spline(self, stamps, quat):
        # Normalize quaternion
        quat /= np.linalg.norm(quat, axis=1)[:, None]

        # Resolve quaternion ambiguities so that quaternion trajectories look
        # smooth. This is recursive so cannot be vectorized.
        for idx, q in enumerate(quat[1:]):
            q_old = quat[idx]
            if np.linalg.norm((-q - q_old)) < np.linalg.norm((q - q_old)):
                q *= -1

        self._quat_spline = csaps(stamps, quat.T, smooth=0.99999)

    @staticmethod
    def from_ros(pose_data, frame_id: Any = None):
        """
        Parameters
        ----------
        pose_data : List[PoseStamped]
            List of ROS PoseStamped messages containing the attitude data
        frame_id : Any
            Optional container to assign an ID to this data
        """

        stamps = []

        # Extract the data from the messages, put in numpy arrays
        position_data = []
        quaternion_data = []
        for p in pose_data:
            stamps.append(p.header.stamp.sec + 1e-9*p.header.stamp.nanosec)
            position_data.append(
                [p.pose.position.x, p.pose.position.y, p.pose.position.z]
            )
            quaternion_data.append(
                [
                    p.pose.orientation.w,
                    p.pose.orientation.x,
                    p.pose.orientation.y,
                    p.pose.orientation.z,
                ]
            )
        stamps = np.array(stamps)
        position_data = np.array(position_data)
        quaternion_data = np.array(quaternion_data)
        return MocapTrajectory(stamps, position_data, quaternion_data, frame_id)

    @staticmethod
    def from_bag(bagfile: str, body_id: str, topic: str = None) -> "MocapTrajectory":
        """
        Loads data directly from a ROS bag file, given the body name you wish to
        extract from. This assumes that a ``vrpn_client_node`` is running and
        publishing a topic of the form ``vrpn_client_node/<body_id>/pose``.

        Parameters
        ----------
        filename : str
            Path to bag file
        body_id : str
            Name of the body to get mocap data for
        topic : str, optional
            exact topic to extract from, by default None

        Returns
        -------
        MocapTrajectory

        Raises
        ------
        ValueError
            If multiple topics for the given body are found
        """

        search_str = f"vrpn_client_node/{body_id}/pose"

        with AnyReader([Path(bagfile)]) as reader:
            topics = reader.topics.keys()

        vrpn_topics = [s for s in topics if search_str in s]

        if len(vrpn_topics) > 1:
            # If more than one matching topic, filter more
            vrpn_topics = [s for s in vrpn_topics if f"{body_id}/" + search_str in s]

        if len(vrpn_topics) > 1:
            raise ValueError(
                "Multiple matching topics found. Please specify"
                + " exactly the right one using the `topic` argument."
            )

        data = bag_to_list(bagfile, vrpn_topics[0])

        return MocapTrajectory.from_ros(data, body_id)

    def plot(self, axs: plt.Axes = None):
        if axs is None:
            fig1, pos_axs = plt.subplots(3, 1, sharex=True)
            fig2, quat_axs = plt.subplots(4, 1, sharex=True)
            fig = [fig1, fig2]
            axs = np.concatenate([pos_axs, quat_axs])
        else:
            fig = [axs[0].figure, axs[3].figure]
            pos_axs = axs[:3]
            quat_axs = axs[3:]

        # Plot position
        pos = self.position(self.stamps)
        pos_axs[0].plot(self.stamps, pos[:, 0])
        pos_axs[1].plot(self.stamps, pos[:, 1])
        pos_axs[2].plot(self.stamps, pos[:, 2])
        pos_axs[2].plot(self.stamps, self.static_mask.astype(int), label="Static")
        pos_axs[0].set_title("Mocap Position Trajectory")
        pos_axs[2].legend()

        # Plot quaternion
        quat = self.quaternion(self.stamps)
        quat_axs[0].plot(self.stamps, quat[:, 0])
        quat_axs[1].plot(self.stamps, quat[:, 1])
        quat_axs[2].plot(self.stamps, quat[:, 2])
        quat_axs[3].plot(self.stamps, quat[:, 3])
        quat_axs[0].set_title("Mocap Quaternion Trajectory")
        quat_axs[0].set_ylim(-1.05, 1.05)
        return fig, axs

    def position(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the position at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            Position data
        """
        return self._pos_spline(stamps, 0).T

    def velocity(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the velocity at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            velocity data
        """
        return self._pos_spline(stamps, 1).T

    def acceleration(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the acceleration at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            acceleration data
        """
        return self._pos_spline(stamps, 2).T

    def accelerometer(self, stamps: np.ndarray, g_a=None) -> np.ndarray:
        """
        Get simuluated accelerometer readings

        Parameters
        ----------
        stamps : float or np.ndarray
            query times
        g_a : List[float], optional
            gravity vector, by default [0, 0, -9.80665]

        Returns
        -------
        ndarray with shape `(len(stamps),3)`
            Accelerometer readings
        """
        if g_a is None:
            g_a = [0, 0, -9.80665]

        a_zwa_a = self._pos_spline(stamps, 2).T
        C_ab = self.rot_matrix(stamps)
        C_ba = np.transpose(C_ab, axes=[0, 2, 1])
        g_a = np.array(g_a).reshape((-1, 1))
        return (C_ba @ (np.expand_dims(a_zwa_a, 2) - g_a)).squeeze()

    def quaternion(self, stamps):
        """
        Get the quaternion at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,4)
            quaternion data
        """
        q = self._quat_spline(stamps, 0).T
        return q / np.linalg.norm(q, axis=1)[:, None]

    def rot_matrix(self, stamps):
        """
        Get the DCM/rotation matrix at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3,3)
            DCM/rotation matrix data
        """

        quat = self.quaternion(stamps)
        return bquat_to_so3(quat)

    def pose_matrix(self, stamps):
        """
        Get the SE(3) pose matrix at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,4,4)
            pose data
        """
        r = self.position(stamps)
        C = self.rot_matrix(stamps)
        T = np.zeros((C.shape[0], 4, 4))
        T[:, :3, :3] = C
        T[:, :3, 3] = r
        T[:, 3, 3] = 1
        return T

    def extended_pose_matrix(self, stamps):
        """
        Get the SE_2(3) extended pose matrix at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,5,5)
            extended pose data
        """
        r = self.position(stamps)
        C = self.rot_matrix(stamps)
        v = self.velocity(stamps)
        T = np.zeros((C.shape[0], 5, 5))
        T[:, :3, :3] = C
        T[:, :3, 3] = v
        T[:, :3, 4] = r
        T[:, 3, 3] = 1
        T[:, 4, 4] = 1
        return T

    def to_pynav(
        self, stamps: np.ndarray, extended_pose: bool = False
    ) -> List[SE23State]:
        """
        Creates pynav ``SE3State`` or ``SE23State`` objects from the trajectory.

        Parameters
        ----------
        stamps : np.ndarray
            query times
        extended_pose : bool, optional
            Whether to return an ``SE3State`` or an ``SE23State`` (if true),
            by default False

        Returns
        -------
        List[SE23State]
            ``pynav`` state objects at the query times
        """
        if extended_pose:
            T = self.extended_pose_matrix(stamps)
            return [
                SE23State(T[i, :, :], stamps[i], self.frame_id)
                for i in range(len(stamps))
            ]

        else:
            T = self.pose_matrix(stamps)

            return [
                SE3State(T[i, :, :], stamps[i], self.frame_id)
                for i in range(len(stamps))
            ]

    def angular_velocity(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the angular velocity at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            angular velocity data
        """
        q = self._quat_spline(stamps, 0)
        q: np.ndarray = q / np.linalg.norm(q, axis=0)
        N = q.shape[1]
        q_dot = np.atleast_2d(self._quat_spline(stamps, 1)).T
        eta = q[0]
        eps = q[1:]

        # TODO: this loop can be vectorized.
        S = np.zeros((N, 3, 4))
        for i in range(N):
            e = eps[:, i].reshape((-1, 1))
            S[i, :, :] = np.hstack((-2 * e, 2 * (eta[i] * np.eye(3) - SO3.wedge(e))))

        omega = (S @ np.expand_dims(q_dot, 2)).squeeze()
        return omega

    def body_velocity(self, stamps: np.ndarray) -> np.ndarray:
        """
        Get the body-frame-resolved translational velocity
        at one or more query times.

        Parameters
        ----------
        stamps : np.ndarray
            Query times

        Returns
        -------
        np.ndarray with shape (N,3)
            body-frame-resolved velocity data
        """
        v_zw_a = self.velocity(stamps)
        v_zw_a = np.expand_dims(v_zw_a, 2)
        C_ab = self.rot_matrix(stamps)
        C_ba = np.transpose(C_ab, axes=[0, 2, 1])
        return (C_ba @ v_zw_a).squeeze().T

    def get_static_mask(
        self, window_size: float, std_dev_threshold: float
    ) -> np.ndarray:
        """
        Detects static moments in the mocap data.

        Parameters
        ----------
        window_size : float
            window size for variance threshold
        std_dev_threshold : float
            threshold value

        Returns
        -------
        np.ndarray
            boolean mask of static moments. True if static at that time.
        """
        # Average mocap frequency
        freq = 1 / ((self.stamps[-1] - self.stamps[0]) / self.stamps.size)

        window_half_width = round(window_size / 2 * freq)
        cov_threshold = std_dev_threshold**2
        is_static = np.zeros((self.stamps.size,), bool)

        for i in range(window_half_width, self.stamps.size - window_half_width):

            pos = self.raw_position[i - window_half_width : i + window_half_width]
            pos_cov = np.cov(pos.T)

            if np.trace(pos_cov) < cov_threshold:
                # Then it is a static
                is_static[i] = True
                if i == window_half_width:
                    is_static[:window_half_width] = True
                elif i == self.stamps.size - window_half_width - 1:
                    is_static[i:] = True

        return is_static

    def is_static(self, stamps: np.ndarray, static_mask=None) -> np.ndarray:
        """
        Returns true or false if the body is detected to be static at time t.
        If ``t`` is a list or numpy array, then it will return a boolean array
        for each point.

        Parameters
        ----------
        t : float or List[float] or numpy.ndarray
            Query stamps
        static_mask : np.ndarray with shape (N,1), optional
            To use a custom static mask to refer to for the static state. This
            must be a boolean array with as many elements as there are stamps
            in this object. If None, then the static mask is computed automatically
            with default parameters. By default None.
        Returns
        -------
        bool or numpy.ndarray
            True if the body is static at time t, False otherwise.
        """

        if static_mask is None:
            static_mask = self.static_mask

        indexes = np.array(range(len(self.stamps)))
        nearest_time_idx = interp1d(
            self.stamps,
            indexes,
            "nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return static_mask[nearest_time_idx(stamps).astype(int)]

    def rotate_body_frame(self, C_mb: np.ndarray) -> "MocapTrajectory":
        """
        Rotates the body frame of the mocap data. The mocap attitude data is
        stored in quaternions corresponding to `C_wm`, which is a rotation matrix
        that rotates the *M*ocap body-frame vectors to world frame vectors `w`.
        I.e.

            v_w = C_wm @ v_m

        The argument, `C_mb` will modify the attitude data such that

            C_new = C_wb = C_wm @ C_mb

        Parameters
        ----------
        C_mb : ndarray with shape `(3,3)`
            A rotation matrix such that C_wb = C_wm @ C_mb
        """
        C_wm = bquat_to_so3(self.raw_quaternion)
        C_wb = C_wm @ C_mb
        q_wb = bso3_to_quat(C_wb)
        return MocapTrajectory(
            self.stamps.copy(),
            self.raw_position.copy(),
            q_wb,
            frame_id=self.frame_id,
        )

    def rotate_world_frame(self, C_wn: np.ndarray) -> "MocapTrajectory":
        """
        Rotates the world frame of the mocap data. The mocap attitude data is
        stored in quaternions corresponding to `C_wm`, which is a rotation matrix
        that rotates the *M*ocap body-frame vectors to *w*orld frame vectors `w`.
        I.e.

            v_w = C_wm @ v_m

        The argument, `C_wn` will modify the attitude data such that it is
        relative to a *n*ew world frame.

            C_new = C_nm = C_wn.T @ C_wm

        Parameters
        ----------
        C_nw : ndarray with shape `(3,3)`
            A rotation matrix such that C_new = C_nm = C_wn.T @ C_wm
        """
        C_wm = bquat_to_so3(self.raw_quaternion)
        C_nm = C_wn.T @ C_wm
        q_nm = bso3_to_quat(C_nm)
        r_zw_w = self.raw_position
        r_zw_n = (C_wn.T @ r_zw_w.T).T
        return MocapTrajectory(self.stamps.copy(), r_zw_n, q_nm, frame_id=self.frame_id)
