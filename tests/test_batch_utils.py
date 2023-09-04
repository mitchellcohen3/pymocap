from pymocap.utils import (
    blog_so3,
    bexp_so3,
    bmv,
    bvee_so3,
    bwedge_so3,
    bquat_to_so3,
    bso3_to_quat,
)
import numpy as np
from pymlg import SO3

np.random.seed(0)


def test_bmv():
    mats = np.random.normal(size=(5, 3, 3))
    vecs = np.random.normal(size=(5, 3))
    out = bmv(mats, vecs)
    out_test = np.array(
        [mats[i, :, :] @ vecs[i, :].reshape((-1, 1)) for i in range(mats.shape[0])]
    ).squeeze()
    assert np.allclose(out, out_test)
    assert out.shape == out_test.shape


def test_bwedge_so3():
    phi = np.random.normal(size=(5, 3))
    Xi = bwedge_so3(phi)
    Xi_test = np.array([SO3.wedge(phi[i, :]) for i in range(phi.shape[0])])
    assert np.allclose(Xi, Xi_test)
    assert Xi.shape == Xi_test.shape


def test_bvee_so3():
    xis = np.random.normal(size=(5, 3))
    Xi = np.array([SO3.wedge(xis[i, :]) for i in range(xis.shape[0])])
    phi = bvee_so3(Xi)
    phi_test = np.array([SO3.vee(Xi[i, :, :]).ravel() for i in range(Xi.shape[0])])
    assert np.allclose(phi, phi_test)
    assert phi.shape == phi_test.shape


def test_bexp_so3():
    phi = np.random.normal(size=(5, 3))
    C = bexp_so3(phi)
    C_test = np.array([SO3.Exp(phi[i, :]) for i in range(phi.shape[0])])
    assert np.allclose(C, C_test)
    assert C.shape == C_test.shape


def test_blog_so3():
    C = np.array([SO3.Exp(np.random.normal(size=(3,))) for i in range(5)])
    phi = blog_so3(C)
    phi_test = np.array([SO3.Log(C[i, :, :]) for i in range(C.shape[0])]).squeeze()
    assert np.allclose(phi, phi_test)
    assert phi.shape == phi_test.shape


def test_blog_bexp_inverse():
    phi = np.random.normal(size=(5, 3))
    C = bexp_so3(phi)
    phi_test = blog_so3(C)
    assert np.allclose(phi, phi_test)
    assert phi.shape == phi_test.shape


def test_bquat():
    q = np.random.normal(size=(5, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    C = bquat_to_so3(q)
    q_test = bso3_to_quat(C)
    C_test = bquat_to_so3(q_test)
    assert np.allclose(C, C_test)
    assert q.shape == q_test.shape


def test_bquat_pymlg():
    q = np.random.normal(size=(5, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    C = bquat_to_so3(q)
    C_test = np.array([SO3.from_quat(q[i, :]) for i in range(q.shape[0])])

    q_test = bso3_to_quat(C_test)
    q_test_pymlg = np.array(
        [SO3.to_quat(C_test[i, :, :]).ravel() for i in range(C_test.shape[0])]
    )
    assert np.allclose(q_test, q_test_pymlg)
    assert np.allclose(C, C_test)
    assert C.shape == C_test.shape


if __name__ == "__main__":
    test_bvee_so3()
