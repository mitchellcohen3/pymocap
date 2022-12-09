import rosbag
import rospy
from typing import List, Any

def bag_to_list(
    bag: rosbag.Bag, topic: str, start_time: float = 0.0, duration: float = None
) -> List[Any]:
    """
    Extracts all the ROS messages from a given topic and returns them as a list.

    Parameters
    ----------
    bag : rosbag.Bag or str
        Bag file as either a rosbag.Bag object or a path to a bag file.
    topic : str
        Topic to extract messages from.
    start_time : float, optional
        Start time after which to extract data, by default 0.0
    duration : float, optional
        Duration of data, after the start time, to extract, by default None. If 
        None, all data after the start time is extracted.

    Returns
    -------
    List[Any]
        List of ROS messages.
    """
    if isinstance(bag, str):
        filename_provided = True
        f = open(bag, "rb")
        bag = rosbag.Bag(f)
    else: 
        filename_provided = False

    bag_start_time = rospy.Time.from_sec(bag.get_start_time() + start_time)

    if duration is None:
        end_time = rospy.Time.from_sec(bag.get_end_time())
    else:
        end_time = rospy.Time.from_sec(
            bag.get_start_time() + start_time + duration
        )

    if filename_provided:
        f.close()

    return [
        msg for _, msg, _ in bag.read_messages(topic, bag_start_time, end_time)
    ]
