from typing import Dict, List, Tuple, Any
import rosbag
import numpy as np
from uwb_ros.msg import RangeStamped
from pynav.types import Measurement, MeasurementModel
import rospy
import pickle
from copy import deepcopy
from dataclasses import dataclass
import matplotlib.pyplot as plt
import itertools
from pynav.lib.models import RangePoseToPose
from .utils import bag_to_list
from .mocap import MocapTrajectory


@dataclass
class Tag:
    id: int
    parent_id: Any
    position: np.ndarray

    def __init__(self, id, parent_id, position):
        self.id = id
        self.parent_id = parent_id
        self.position = np.array(position).ravel()


class RangeData:
    def __init__(
        self,
        stamps,
        range,
        from_id,
        to_id,
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
        self.range = np.array(range).ravel()
        self.from_id = np.array(from_id).ravel()
        self.to_id = np.array(to_id).ravel()
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
    def from_ros(range_data: List[RangeStamped]):

        stamps = []
        range = []
        from_id = []
        to_id = []
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
            range.append(r.range)
            from_id.append(r.from_id)
            to_id.append(r.to_id)
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
            range,
            from_id,
            to_id,
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
    def from_bag(bag, topic: str):
        height_msgs = bag_to_list(bag, topic)
        return RangeData.from_ros(height_msgs)

    def by_pair(self, from_id, to_id):
        match_mask = np.logical_and(
            self.from_id == from_id, self.to_id == to_id
        )
        return RangeData(
            self.stamps[match_mask],
            self.range[match_mask],
            self.from_id[match_mask],
            self.to_id[match_mask],
            self.tx1[match_mask],
            self.rx1[match_mask],
            self.tx2[match_mask],
            self.rx2[match_mask],
            self.tx3[match_mask],
            self.rx3[match_mask],
            self.fpp1[match_mask],
            self.fpp2[match_mask],
            self.rxp1[match_mask],
            self.rxp2[match_mask],
            self.std1[match_mask],
            self.std2[match_mask],
        )

    def to_pynav(
        self, tags: List[Tag], variance=0.1**2, state_id: Any = None
    ):
        tag_dict = {t.id: t for t in tags}
        measurements = []

        model_dict = {}

        for i, stamp in enumerate(self.stamps):
            from_tag = tag_dict[self.from_id[i]]
            to_tag = tag_dict[self.to_id[i]]

            if (from_tag.id, to_tag.id) not in model_dict:
                model_dict[(from_tag.id, to_tag.id)] = RangePoseToPose(
                    from_tag.position,
                    to_tag.position,
                    from_tag.parent_id,
                    to_tag.parent_id,
                    variance,  # TODO. use modelled variance instead.
                )

            model = model_dict[(from_tag.id, to_tag.id)]

            measurements.append(
                Measurement(self.range[i], stamp, model, state_id=state_id)
            )

        return measurements

    def plot(
        self, mocaps: List[MocapTrajectory] = None, tags: List[Tag] = None
    ):

        pairs = set(zip(self.from_id, self.to_id))

        if len(pairs) > 20:
            raise ValueError(
                "Too many pairs to plot. This code needs to be improved."
            )

        num_cols = 4
        num_rows = int(np.ceil(len(pairs) / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, sharex=True)
        axes = axes.ravel()

        if mocaps is not None:
            pose_dict = {m.frame_id: m.pose_matrix(self.stamps) for m in mocaps}

        if tags is not None:
            tag_dict = {t.id: t for t in tags}

        for i, pair in enumerate(pairs):
            ax = axes[i]
            data = self.by_pair(*pair)
            ax.scatter(data.stamps, data.range, s=1)
            ax.set_title(f"{pair[0]} to {pair[1]}")

            if mocaps is not None and tags is not None:
                tag1 = tag_dict[pair[0]]
                tag2 = tag_dict[pair[1]]
                pose1 = pose_dict[tag1.parent_id]
                pose2 = pose_dict[tag2.parent_id]
                range = _get_range(
                    pose1[:, :3, 3],
                    pose1[:, :3, :3],
                    pose2[:, :3, 3],
                    pose2[:, :3, :3],
                    tag1.position,
                    tag2.position,
                )
                ax.plot(self.stamps, range, color="r")

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

    def to_pynav(
        self, pair_models: Dict[Tuple, MeasurementModel], state_id: Any
    ) -> List[Measurement]:
        """

        Parameters
        ----------
        pair_models : Dict[Tuple, MeasurementModel]
            Dictionary of measurement model to associate with each range pair.
        state_id : Any
            State ID to assign to each ``Measurement``.

        Returns
        -------
        List[Measurement]
            The measurements in the form of a list of ``Measurement`` objects.
        """
        data = []
        for pair in self.pairs:
            t, r = self.by_pair(*pair)
            model = pair_models[pair]
            for i in range(len(t)):
                data.append(
                    Measurement(
                        r[i],
                        t[i],
                        model,
                        state_id=state_id,
                    )
                )
        return data


def _get_range(pos1, att1, pos2, att2, tag_pos1, tag_pos2):
    r_1w_a = pos1
    C_a1 = att1
    r_2w_a = pos2
    C_a2 = att2
    r_t1_1 = tag_pos1
    r_t2_2 = tag_pos2
    r_t1t2_a: np.ndarray = (C_a1 @ r_t1_1 + r_1w_a) - (C_a2 @ r_t2_2 + r_2w_a)
    return np.linalg.norm(r_t1t2_a, axis=1)


class RangeCorrector:
    """

    Object to retrieve and correct range measurements.
    """

    _c = 299702547  # speed of light
    _dwt_to_ns = 1e9 * (1.0 / 499.2e6 / 128.0)  # DW time unit to nanoseconds

    def __init__(self):
        """
        Constructor
        """
        # Retrieve pre-determined calibration results
        with open("multinav/calib_results.pickle", "rb") as pickle_file:
            calib_results = pickle.load(pickle_file)

        self.delays = calib_results["delays"]
        self.bias_spl = calib_results["bias_spl"]
        self.std_spl = calib_results["std_spl"]

    def get_corrected_range(self, uwb_msg: RangeStamped) -> RangeStamped:
        """
        Extracts and corrects the range measurement and
        associated information.

        PARAMETERS:
        -----------
        uwb_msg: RangeStamped
            One instance of UWB data.

        RETURNS:
        --------
         dict: Dictionary with 4 fields.
            from_id: int
                ID of initiating tag.
            to_id: int
                ID of target tag.
            range: float
                Corrected range measurement.
            std: float
                Standard deviation of corrected range measurement.
        """
        # Get tag IDs
        from_id = uwb_msg.from_id
        to_id = uwb_msg.to_id
        # Get timestamps
        tx1 = uwb_msg.tx1 * self._dwt_to_ns
        rx1 = uwb_msg.rx1 * self._dwt_to_ns
        tx2 = uwb_msg.tx2 * self._dwt_to_ns
        rx2 = uwb_msg.rx2 * self._dwt_to_ns
        tx3 = uwb_msg.tx3 * self._dwt_to_ns
        rx3 = uwb_msg.rx3 * self._dwt_to_ns

        # Correct clock wrapping
        rx2, rx3 = self._unwrap_ts(tx1, rx2, rx3)
        tx2, tx3 = self._unwrap_ts(rx1, tx2, tx3)

        # Compute time intervals
        Ra1 = rx2 - tx1
        Ra2 = rx3 - rx2
        Db1 = tx2 - rx1
        Db2 = tx3 - tx2

        # Get antenna delays
        delay_0 = self.delays[from_id]
        delay_1 = self.delays[to_id]

        # Correct time intervals for antenna delays
        Ra1 += delay_0
        Db1 -= delay_1

        # Get power
        fpp1 = uwb_msg.fpp1
        fpp2 = uwb_msg.fpp2

        # Implement lifting function
        fpp1_lift = self.lift(fpp1)
        fpp2_lift = self.lift(fpp2)

        # Get average lifted power
        fpp_lift_avg = 0.5 * (fpp1_lift + fpp2_lift)

        # Power-induced bias
        bias = self.bias_spl(fpp_lift_avg)

        # Compute range measurement
        range = self._compute_range(Ra1, Ra2, Db1, Db2, bias)

        # Get standard deviation of measurement
        std = float(self.std_spl(fpp_lift_avg))

        msg_new = deepcopy(uwb_msg)
        msg_new.range = range
        msg_new.covariance = std**2
        return msg_new

    def _unwrap_ts(self, ts1, ts2, ts3):
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
        max_time_ns = 2**32 * self._dwt_to_ns

        if ts2 < ts1:
            ts2 += max_time_ns
            ts3 += max_time_ns
        if ts3 < ts2:
            ts3 += max_time_ns

        return ts2, ts3

    @staticmethod
    def lift(x, alpha=-82):
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

    def _compute_range(self, Ra1, Ra2, Db1, Db2, bias):
        """
        Compute the bias-corrected range measurement.

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
        return 0.5 * self._c / 1e9 * (Ra1 - (Ra2 / Db2) * Db1) - bias
