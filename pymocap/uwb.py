from typing import List, Any, Tuple
import numpy as np
from pynav.types import Measurement
import pickle
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pynav.lib.models import RangePoseToPose
from .utils import bag_to_list, bmv
from .mocap import MocapTrajectory

try: 
    from uwb_ros.msg import RangeStamped
except ImportError:
    pass


SPEED_OF_LIGHT = 299702547  # speed of light in m/s
DWT_TO_NS = 1e9 * (1.0 / 499.2e6 / 128.0)  # DW time unit to nanoseconds


@dataclass
class Tag:
    id: int
    parent_id: Any
    position: np.ndarray
    antenna_delay: float

    def __init__(self, id, parent_id, position, antenna_delay):
        self.id = id
        self.parent_id = parent_id
        self.position = np.array(position).ravel()
        self.antenna_delay = antenna_delay


class RangeData:
    """
    Class for storing UWB range data.
    """
    def __init__(
        self,
        stamps,
        range_,
        from_id,
        to_id,
        covariance,
        tx1,
        rx1,
        tx2,
        rx2,
        tx3,
        rx3,
        fpp1,
        fpp2,
        rxp1,
        rxp2,
        std1,
        std2,
    ):
        self.stamps = np.array(stamps).ravel()
        self.range = np.array(range_).ravel()
        self.from_id = np.array(from_id).ravel()
        self.to_id = np.array(to_id).ravel()
        self.covariance = np.array(covariance).ravel()
        self.tx1 = np.array(tx1).ravel()
        self.rx1 = np.array(rx1).ravel()
        self.tx2 = np.array(tx2).ravel()
        self.rx2 = np.array(rx2).ravel()
        self.tx3 = np.array(tx3).ravel()
        self.rx3 = np.array(rx3).ravel()
        self.fpp1 = np.array(fpp1).ravel()
        self.fpp2 = np.array(fpp2).ravel()
        self.rxp1 = np.array(rxp1).ravel()
        self.rxp2 = np.array(rxp2).ravel()
        self.std1 = np.array(std1).ravel()
        self.std2 = np.array(std2).ravel()

    @staticmethod
    def from_ros(range_data):
        """
        Create a RangeData object from a list of ROS ``RangeStamped`` messages.

        Parameters
        ----------
        range_data : List[RangeStamped]
            List of ROS ``RangeStamped`` messages from `uwb_ros` package.

        Returns
        -------
        RangeData
            RangeData object.
        """


        stamps = []
        range_ = []
        from_id = []
        to_id = []
        covariance = []
        tx1 = []
        rx1 = []
        tx2 = []
        rx2 = []
        tx3 = []
        rx3 = []
        fpp1 = []
        fpp2 = []
        rxp1 = []
        rxp2 = []
        std1 = []
        std2 = []

        for r in range_data:
            stamps.append(r.header.stamp.to_sec())
            range_.append(r.range)
            from_id.append(r.from_id)
            to_id.append(r.to_id)
            covariance.append(r.covariance)
            tx1.append(r.tx1)
            rx1.append(r.rx1)
            tx2.append(r.tx2)
            rx2.append(r.rx2)
            tx3.append(r.tx3)
            rx3.append(r.rx3)
            fpp1.append(r.fpp1)
            fpp2.append(r.fpp2)
            rxp1.append(r.rxp1)
            rxp2.append(r.rxp2)
            std1.append(r.std1)
            std2.append(r.std2)

        out = RangeData(
            stamps,
            range_,
            from_id,
            to_id,
            covariance,
            tx1,
            rx1,
            tx2,
            rx2,
            tx3,
            rx3,
            fpp1,
            fpp2,
            rxp1,
            rxp2,
            std1,
            std2,
        )
        return out

    @staticmethod
    def from_bag(bag, topic: str) -> "RangeData":
        """
        Create a RangeData object from a ROS bag.

        Parameters
        ----------
        bag : rosbag.Bag or str
            rosbag.Bag object or path to bag file.
        topic : str
            topic containing range data. Must be of type `uwb_ros/RangeStamped`.

        Returns
        -------
        RangeData
            RangeData object.
        """
        height_msgs = bag_to_list(bag, topic)
        return RangeData.from_ros(height_msgs)

    def __getitem__(self, slc):
        return RangeData(
            self.stamps[slc],
            self.range[slc],
            self.from_id[slc],
            self.to_id[slc],
            self.covariance[slc],
            self.tx1[slc],
            self.rx1[slc],
            self.tx2[slc],
            self.rx2[slc],
            self.tx3[slc],
            self.rx3[slc],
            self.fpp1[slc],
            self.fpp2[slc],
            self.rxp1[slc],
            self.rxp2[slc],
            self.std1[slc],
            self.std2[slc],
        )

    def by_pair(self, from_id :int, to_id:int) -> "RangeData":
        """
        Get a RangeData object containing only the measurements between the
        specified pair of tags.

        Parameters
        ----------
        from_id : int
            The ID of the intiating tag.
        to_id : int
            The ID of the receiving tag.

        Returns
        -------
        RangeData
            RangeData object
        """
        match_mask = np.logical_and(
            self.from_id == from_id, self.to_id == to_id
        )
        return self[match_mask]

    def to_pynav(self, tags: List[Tag], variance: float = None, state_id: Any = None) -> List[Measurement]:
        """
        Convert to a list of `pynav` measurements.

        Parameters
        ----------
        tags : List[Tag]
            Information about the tags
        variance : float, optional
            If specified, overrides the variance for all measurements. Otherwise
            the variance from the bag file or the calibration is used.
        state_id : Any, optional
            optional identifier to add to the measurement, by default None

        Returns
        -------
        List[Measurement]
            List of `pynav` measurements.
        """

        tag_dict = {t.id: t for t in tags}
        measurements = []

        for i, stamp in enumerate(self.stamps):
            from_tag = tag_dict[self.from_id[i]]
            to_tag = tag_dict[self.to_id[i]]

            if variance is not None:
                v = variance
            else:
                v = self.covariance[i]

            model = RangePoseToPose(
                from_tag.position,
                to_tag.position,
                from_tag.parent_id,
                to_tag.parent_id,
                v,
            )


            measurements.append(
                Measurement(self.range[i], stamp, model, state_id=state_id)
            )

        return measurements

    def plot(
        self, mocaps: List[MocapTrajectory] = None, tags: List[Tag] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot the range data.

        Parameters
        ----------
        mocaps : List[MocapTrajectory], optional
            Mocap trajectories for all the bodies involved in the data.
            If not provided, no ground truth will be plotted.
        tags : List[Tag], optional
            Tag info for all the tags. If not provided, no ground truth 
            will be plotted.

        Returns
        -------
        Tuple[plt.Figure, List[plt.Axes]]
            Figure and axes objects.

        Raises
        ------
        ValueError
            If there are more than 20 pairs of tags in the data.
        """

        pairs = set(zip(self.from_id, self.to_id))

        if len(pairs) > 20:
            raise ValueError(
                "Too many pairs to plot. This code needs to be improved."
            )

        num_cols = 4
        num_rows = int(np.ceil(len(pairs) / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, sharex=True)
        axes = axes.ravel()

        if tags is not None:
            tag_dict = {t.id: t for t in tags}

        for i, pair in enumerate(pairs):
            ax = axes[i]
            data = self.by_pair(*pair)
            ax.scatter(data.stamps, data.range, s=1)
            ax.set_title(f"{pair[0]} to {pair[1]}")

            if mocaps is not None and tags is not None:
                pose_dict = {m.frame_id: m.pose_matrix(data.stamps) for m in mocaps}
                tag1 = tag_dict[pair[0]]
                tag2 = tag_dict[pair[1]]
                pose1 = pose_dict[tag1.parent_id]
                pose2 = pose_dict[tag2.parent_id]
                range_ = _get_range_from_poses(
                    pose1[:, :3, 3],
                    pose1[:, :3, :3],
                    pose2[:, :3, 3],
                    pose2[:, :3, :3],
                    tag1.position,
                    tag2.position,
                )
                ax.plot(data.stamps, range_, color="r")
                three_sigma = 3 * np.sqrt(data.covariance)
                ax.fill_between(
                    data.stamps,
                    range_ - three_sigma,
                    range_ + three_sigma,
                    alpha=0.3,
                    color="r",
                )

        fig.suptitle("Range Data")
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        fig.supxlabel("Time (s)")
        fig.supylabel("Range (m)")

        return fig, axes

    def remove_outliers(
        self, mocaps: List[MocapTrajectory], tags: List[Tag], max_error: float
    ) -> "RangeData":
        """
        Remove outliers from the range data.

        Parameters
        ----------
        mocaps : List[MocapTrajectory]
            Mocap trajectories for all the bodies involved in the data.
        tags : List[Tag]
            Tag info for all the tags.
        max_error : float
            Maximum error in meters. If the range error is larger than this, the
            measurement is removed.

        Returns
        -------
        RangeData
            New range data object with outliers removed.
        """
    

        pose_dict = {m.frame_id: m.pose_matrix(self.stamps) for m in mocaps}
        tag_dict = {t.id: t for t in tags}
        pose1 = np.array(
            [
                pose_dict[tag_dict[t].parent_id][i]
                for i, t in enumerate(self.from_id)
            ]
        )

        pose2 = np.array(
            [
                pose_dict[tag_dict[t].parent_id][i]
                for i, t in enumerate(self.to_id)
            ]
        )

        tag_position1 = np.array([tag_dict[t].position for t in self.from_id])
        tag_position2 = np.array([tag_dict[t].position for t in self.to_id])

        range_ = _get_range_from_poses(
            pose1[:, :3, 3],
            pose1[:, :3, :3],
            pose2[:, :3, 3],
            pose2[:, :3, :3],
            tag_position1,
            tag_position2,
        )

        is_outlier_mask = np.abs(range_ - self.range) > max_error
        is_outlier_mask = np.logical_or(is_outlier_mask, self.range < 0)
        return self[~is_outlier_mask]


    def apply_calibration(self, tags: List[Tag]) -> "RangeData":
        """
        Apply power-correlated bias correction to the range data.

        Parameters
        ----------
        tags : List[Tag]
            Tag info for all the tags. Critically, these dataclasses must contain
            a valid ``antenna_delay`` attribute.

        Returns
        -------
        RangeData
            New range data object with bias correction applied.
        """

        # Retrieve pre-determined calibration results
        with open("pymocap/calib_results.pickle", "rb") as pickle_file:
            calib_results = pickle.load(pickle_file)

        delays = {t.id: t.antenna_delay for t in tags}

        bias_spline = calib_results["bias_spl"]
        std_dev_spline = calib_results["std_spl"]

        # Get timestamps
        tx1 = self.tx1 * DWT_TO_NS
        rx1 = self.rx1 * DWT_TO_NS
        tx2 = self.tx2 * DWT_TO_NS
        rx2 = self.rx2 * DWT_TO_NS
        tx3 = self.tx3 * DWT_TO_NS
        rx3 = self.rx3 * DWT_TO_NS

        # Correct clock wrapping
        rx2, rx3 = _unwrap_ts(tx1, rx2, rx3)
        tx2, tx3 = _unwrap_ts(rx1, tx2, tx3)

        # Compute time intervals
        Ra1 = rx2 - tx1
        Ra2 = rx3 - rx2
        Db1 = tx2 - rx1
        Db2 = tx3 - tx2

        # Get antenna delays
        # TODO. should be a better way to vectorize this.
        delay_0 = np.array([delays[i] for i in self.from_id])
        delay_1 = np.array([delays[i] for i in self.to_id])

        # Correct time intervals for antenna delays
        Ra1 += delay_0
        Db1 -= delay_1

        # Get power
        fpp1 = self.fpp1
        fpp2 = self.fpp2

        # Implement lifting function
        fpp1_lift = _lift(fpp1)
        fpp2_lift = _lift(fpp2)

        # Get average lifted power
        fpp_lift_avg = 0.5 * (fpp1_lift + fpp2_lift)

        # Power-induced bias
        bias = bias_spline(fpp_lift_avg)

        # Compute range measurement
        range_ = _compute_range_dstwr(Ra1, Ra2, Db1, Db2, bias)

        # Get standard deviation of measurement
        std = std_dev_spline(fpp_lift_avg)

        # Create new RangeData object
        new_range_data = RangeData(
            self.stamps,
            range_,
            self.from_id,
            self.to_id,
            std**2,
            self.tx1,
            self.rx1,
            self.tx2,
            self.rx2,
            self.tx3,
            self.rx3,
            self.fpp1,
            self.fpp2,
            self.rxp1,
            self.rxp2,
            self.std1,
            self.std2,
        )
        return new_range_data


def _get_range_from_poses(pos1, att1, pos2, att2, tag_pos1, tag_pos2):
    r_1w_a = pos1
    C_a1 = att1
    r_2w_a = pos2
    C_a2 = att2
    r_t1_1 = tag_pos1
    if r_t1_1.size == 3:
        r_t1_1 = r_t1_1.ravel()
        r_t1_1 = np.tile(r_t1_1, [C_a1.shape[0],1])

        
    r_t2_2 = tag_pos2
    if r_t2_2.size == 3:
        r_t2_2 = r_t2_2.ravel()
        r_t2_2 = np.tile(r_t2_2, [C_a2.shape[0],1])





    r_t1t2_a: np.ndarray = (bmv(C_a1,r_t1_1) + r_1w_a) - (bmv(C_a2, r_t2_2) + r_2w_a)
    return np.linalg.norm(r_t1t2_a, axis=1)


def _unwrap_ts(ts1, ts2, ts3):
    """
    Corrects the UWB-module's clock unwrapping.

    PARAMETERS:
    -----------
    ts1: int
        First timestamp in a sequence of timestamps registered
        on the same clock.
    ts2: int
        Second timestamp in a sequence of timestamps registered
        on the same clock.
    ts3: int
        Third timestamp in a sequence of timestamps registered
        on the same clock.

    RETURNS:
    --------
    ts2: int
        Unwrapped second timestamp in a sequence of timestamps
        registered on the same clock.
    ts3: int
        Unwrapped third timestamp in a sequence of timestamps
        registered on the same clock.
    """
    # The timestamps are registered as type uint32.
    max_time_ns = 2**32 * DWT_TO_NS

    ts2_is_wrapped = ts2 < ts1
    ts2[ts2_is_wrapped] += max_time_ns
    ts3[ts2_is_wrapped] += max_time_ns

    ts3_is_wrapped = ts3 < ts2
    ts3[ts3_is_wrapped] += max_time_ns

    return ts2, ts3


def _lift(x, alpha=-82):
    """
    Lifting function for better visualization and calibration.
    Based on Cano, J., Pages, G., Chaumette, E., & Le Ny, J. (2022). Clock
                and Power-Induced Bias Correction for UWB Time-of-Flight Measurements.
                IEEE Robotics and Automation Letters, 7(2), 2431-2438.
                https://doi.org/10.1109/LRA.2022.3143202

    PARAMETERS:
    -----------
    x: np.array(n,1)
        Input to lifting function. Received Power in dBm in this context.
    alpha: scalar
        Centering parameter. Default: -82 dBm.

    RETURNS:
    --------
    np.array(n,1)
        Array of lifted received power.
    """
    return 10 ** ((x - alpha) / 10)


def _compute_range_dstwr(Ra1, Ra2, Db1, Db2, bias):
    """
    Compute the bias-corrected range measurement using double-sided TWR
    and a bias correction.

    PARAMETERS:
    -----------
    Ra1: int
        rx2 - rx1.
    Ra2: int
        rx3 - rx2.
    Db1: int
        tx2 - rx1.
    Db2: int
        tx3 - tx2.
    bias: float
        Estimated power-induced bias.

    RETURNS:
    --------
    float
        Bias-corrected range measurement.
    """
    return (0.5 * SPEED_OF_LIGHT / 1e9) * (Ra1 - (Ra2 / Db2) * Db1) - bias
