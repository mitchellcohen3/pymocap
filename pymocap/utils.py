import rosbag
import rospy
from typing import List, Any
import numpy as np


def bag_to_list(
    bagfile: rosbag.Bag,
    topic: str,
    start_time: float = 0.0,
    duration: float = None,
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
    if not isinstance(bagfile, rosbag.Bag):
        bag = rosbag.Bag(bagfile, "r")
    else:
        bag = bagfile

    bag_start_time = rospy.Time.from_sec(bag.get_start_time() + start_time)

    if duration is None:
        end_time = rospy.Time.from_sec(bag.get_end_time())
    else:
        end_time = rospy.Time.from_sec(bag.get_start_time() + start_time + duration)

    out = [msg for _, msg, _ in bag.read_messages(topic, bag_start_time, end_time)]

    if not isinstance(bagfile, rosbag.Bag):
        bag.close()

    return out


def bmv(matrices, vectors) -> np.ndarray:
    """
    Batch matrix vector multiplication
    """
    return np.matmul(matrices, vectors[:, :, None]).squeeze()


def bmm(matrices_a, matrices_b) -> np.ndarray:
    """
    Batch matrix multiplication
    """
    return np.matmul(matrices_a, matrices_b)


def beye(dim: int, num: int) -> np.ndarray:
    """
    Batch identity matrix
    """
    return np.tile(np.eye(dim), (num, 1, 1))


def bwedge_so3(phi: np.ndarray) -> np.ndarray:
    """
    Batch wedge for SO(3). phi is provided as a [N x 3] ndarray
    """

    if phi.shape[1] != 3:
        raise ValueError("phi must have shape ({},) or (N,{})".format(3, 3))

    Xi = np.zeros((phi.shape[0], 3, 3))
    Xi[:, 0, 1] = -phi[:, 2]
    Xi[:, 0, 2] = phi[:, 1]
    Xi[:, 1, 0] = phi[:, 2]
    Xi[:, 1, 2] = -phi[:, 0]
    Xi[:, 2, 0] = -phi[:, 1]
    Xi[:, 2, 1] = phi[:, 0]

    return Xi


def bvee_so3(Xi: np.ndarray):
    """
    Batch vee for SO(3). Xi is provided as a [N x 3 x 3] ndarray
    """

    if Xi.shape[1] != 3 or Xi.shape[2] != 3:
        raise ValueError("Xi must have shape (N,3,3)")

    phi = np.zeros((Xi.shape[0], 3))
    phi[:, 0] = -Xi[:, 1, 2]
    phi[:, 1] = Xi[:, 0, 2]
    phi[:, 2] = -Xi[:, 0, 1]

    return phi


def bouter(a, b):
    """
    Batch outer product
    """
    return np.einsum("...i,...j->...ij", a, b)


def btrace(matrices):
    """
    Batch trace
    """
    return np.trace(matrices, axis1=1, axis2=2)


def bexp_so3(phi: np.ndarray):
    """
    Batch exponential map for SO(3). phi is provided as a [N x 3] ndarray
    """

    if phi.shape[1] != 3:
        raise ValueError("phi must have shape (N,3)")

    out = np.zeros((phi.shape[0], 3, 3))
    angle = np.linalg.norm(phi, axis=1)

    # Near phi==0, use first order Taylor expansion
    small_angle_mask = np.isclose(angle, 0.0, atol=1e-7)

    if np.any(small_angle_mask):
        out[small_angle_mask] = beye(3, np.sum(small_angle_mask)) + bwedge_so3(
            phi[small_angle_mask]
        )

    # Otherwise...
    large_angle_mask = np.logical_not(small_angle_mask)

    if np.any(large_angle_mask):
        large_angles = angle[large_angle_mask].reshape((-1, 1))
        axis = phi[large_angle_mask] / large_angles
        s = np.sin(large_angles).reshape((-1, 1, 1))
        c = np.cos(large_angles).reshape((-1, 1, 1))

        A = c * beye(3, np.sum(large_angle_mask))
        B = (1.0 - c) * bouter(axis, axis)
        C = s * bwedge_so3(axis)

        out[large_angle_mask] = A + B + C

    return out


def blog_so3(C: np.ndarray):
    if C.shape[1] != 3 or C.shape[2] != 3:
        raise ValueError("C must have shape (N,3,3)")

    phi = np.zeros((C.shape[0], 3))

    # The cosine of the rotation angle is related to the trace of C
    # Clamp to its proper domain to avoid NaNs from rounding errors
    cos_angle = np.clip((0.5 * btrace(C) - 0.5).ravel(), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # Near phi==0, use first order Taylor expansion
    small_angle_mask = np.isclose(angle, 0.0, atol=1e-7)

    if np.any(small_angle_mask):
        phi[small_angle_mask, :] = bvee_so3(C[small_angle_mask]) - beye(
            3, np.sum(small_angle_mask)
        )

    # Otherwise...
    large_angle_mask = np.logical_not(small_angle_mask)

    if np.any(large_angle_mask):
        large_angles = angle[large_angle_mask].reshape((-1, 1))
        sin_angle = np.sin(large_angles)
        Xi = (0.5 * large_angles / sin_angle).reshape((-1, 1, 1)) * (
            C[large_angle_mask] - np.transpose(C[large_angle_mask], [0, 2, 1])
        )
        phi[large_angle_mask, :] = bvee_so3(Xi)

    return phi


def bquat_to_so3(quat: np.ndarray, ordering="wxyz"):
    """
    Form a rotation matrix from a unit length quaternion.
    Valid orderings are 'xyzw' and 'wxyz'.
    """

    if not np.allclose(np.linalg.norm(quat, axis=1), 1.0):
        raise ValueError("Quaternions must be unit length")

    if ordering == "wxyz":
        eta = quat[:, 0]
        eps = quat[:, 1:]
    elif ordering == "xyzw":
        eta = quat[:, 3]
        eps = quat[:, 0:3]
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'. ")
    eta = eta.reshape((-1, 1, 1))
    eps = eps.reshape((-1, 3, 1))
    eps_T = np.transpose(eps, [0, 2, 1])
    return (
        (1 - 2 * eps_T @ eps) * beye(3, quat.shape[0])
        + 2 * eps @ eps_T
        + 2 * eta * bwedge_so3(eps.squeeze())
    )


def bso3_to_quat(C: np.ndarray, order="wxyz"):
    """
    Returns the quaternion corresponding to DCM C.

    Parameters
    ----------
    C : ndarray with shape (N,3,3)
        DCM/rotation matrix to convert.
    order : str, optional
        quaternion element order "xyzw" or "wxyz", by default "wxyz"

    Returns
    -------
    ndarray with shape (4,1)
            quaternion representation of C

    Raises
    ------
    ValueError
        if `C` does not have shape (3,3)
    ValueError
        if `order` is not "xyzw" or "wxyz"
    """

    if C.shape[1] != 3 or C.shape[2] != 3:
        raise ValueError("C must have shape (N,3,3)")

    eta = 0.5 * (btrace(C) + 1) ** 0.5
    # eps = -np.array(
    #     [C[1, 2] - C[2, 1], C[2, 0] - C[0, 2], C[0, 1] - C[1, 0]]
    # ) / (4 * eta)
    eps = np.zeros((C.shape[0], 3))
    eps[:, 0] = C[:, 2, 1] - C[:, 1, 2]
    eps[:, 1] = C[:, 0, 2] - C[:, 2, 0]
    eps[:, 2] = C[:, 1, 0] - C[:, 0, 1]
    eps = eps / (4 * eta).reshape((-1, 1))

    q = np.zeros((C.shape[0], 4))
    if order == "wxyz":
        q[:, 0] = eta
        q[:, 1:] = eps
    elif order == "xyzw":
        q[:, 0:3] = eps
        q[:, 3] = eta
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'. ")

    return q
