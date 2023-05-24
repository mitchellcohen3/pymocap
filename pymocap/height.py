from typing import Dict, List, Tuple, Any
import numpy as np
from .mocap import MocapTrajectory
import matplotlib.pyplot as plt
from .utils import bag_to_list

try:
    from pynav import plot_meas, Measurement
    from pynav.lib import Altitude
except ImportError:
    class Measurement:
        def __init__(self, *args, **kwargs):
            raise ImportError("pynav not installed")
        
    class Altitude:
        def __init__(self, *args, **kwargs):
            raise ImportError("pynav not installed")
    
    def plot_meas(*args, **kwargs):
        raise ImportError("pynav not installed")


class HeightData:
    """
    Container for height data from a 1D range sensor.
    """

    def __init__(self, stamps: np.ndarray, height: np.ndarray):
        self.stamps = stamps
        self.height = height

    @staticmethod
    def from_ros(height_data) -> "HeightData":
        """
        Get height data from ROS messages.

        Parameters
        ----------
        height_data : List[Range]
            List of ROS Range messages.

        Returns
        -------
        HeightData
            a HeightData object.
        """
        height = []
        stamps = []
        for h in height_data:
            stamps.append(h.header.stamp.to_sec())
            height.append(h.range)

        stamps = np.array(stamps)
        height = np.array(height)

        return HeightData(stamps, height)

    @staticmethod
    def from_bag(bag, topic: str):
        height_msgs = bag_to_list(bag, topic)
        return HeightData.from_ros(height_msgs)

    def to_pynav(
        self, state_id=None, variance=0.1**2, minimum=0.4, bias=0.0
    ) -> List[Measurement]:
        """
        Converts the distance data to a list of pynav Measurement objects.

        Parameters
        ----------
        state_id : Any, optional
            optional state ID to add to all the Measurement objects, by default None
        variance : float, optional
            Variance of the measurement error, by default 0.1**2
        minimum : float, optional
            minimum value of the measurement for it to be valid, by default 0.4
        bias : float, optional
            sensor bias to be subtracted from the reading, by default 0.0

        Returns
        -------
        List[Measurement]
            pynav Measurement objects with an Altitude model
        """
        data = []
        model = Altitude(variance, minimum=minimum, bias=bias)
        for i in range(len(self.stamps)):

            data.append(
                Measurement(
                    self.height[i],
                    self.stamps[i],
                    model,
                    state_id=state_id,
                )
            )
        return data

    def plot(
        self,
        mocap: MocapTrajectory = None,
        axs: plt.Axes = None,
        variance=0.1**2,
        minimum=0.4,
        bias=0.0,
    ):
        """
        Plots the height measurements against ground truth.

        Parameters
        ----------
        mocap: MocapTrajectory, optional
            ground truth trajectory, by default None
        variance : float, optional
            Variance of the measurement error, by default 0.1**2
        minimum : float, optional
            minimum value of the measurement for it to be valid, by default 0.4
        bias : float, optional
            sensor bias to be subtracted from the reading, by default 0.0
        """

        if mocap is not None:
            height_meas = self.to_pynav(variance=variance, minimum=minimum, bias=bias)
            height_stamps = [meas.stamp for meas in height_meas]
            x_true = mocap.to_pynav(height_stamps)
            fig, axs = plot_meas(height_meas, x_true, axs=axs)

        else:
            if axs is None:
                fig, axs = plt.subplots(1, 1)
            else:
                fig = axs.get_figure()

            axs.plot(self.stamps, self.height)

        if isinstance(axs, np.ndarray):
            axs = axs[0]

        axs.set_xlabel("Time (s)")
        axs.set_ylabel("Height (m)")
        axs.set_title("Height Sensor")
        return fig, axs
