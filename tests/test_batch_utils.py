from pymocap.utils import blog_so3, bexp_so3, bmv, bvee_so3, bwedge_so3
import numpy as np
from pylie import SO3
np.random.seed(0)

def test_bmv():
    mats = np.random.normal(size=(5, 3, 3))
    vecs = np.random.normal(size=(5, 3))
    out = bmv(mats, vecs)
    out_test = np.array(
        [
            mats[i, :, :] @ vecs[i, :].reshape((-1, 1))
            for i in range(mats.shape[0])
        ]
    ).squeeze()
    assert np.allclose(out, out_test)
    assert out.shape == out_test.shape

def test_bwedge_so3():
    phi = np.random.normal(size=(5, 3))
    Xi = bwedge_so3(phi)
    Xi_test = np.array(
        [
            SO3.wedge(phi[i, :])
            for i in range(phi.shape[0])
        ]
    )
    assert np.allclose(Xi, Xi_test)
    assert Xi.shape == Xi_test.shape

def test_bvee_so3():
    Xi = np.random.normal(size=(5, 3, 3))
    phi = bvee_so3(Xi)
    phi_test = np.array(
        [
            SO3.vee(Xi[i, :, :]).ravel()
            for i in range(Xi.shape[0])
        ]
    )
    assert np.allclose(phi, phi_test)
    assert phi.shape == phi_test.shape

def test_bexp_so3():
    phi = np.random.normal(size=(5, 3))
    C = bexp_so3(phi)
    C_test = np.array(
        [
            SO3.Exp(phi[i, :])
            for i in range(phi.shape[0])
        ]
    )
    assert np.allclose(C, C_test)
    assert C.shape == C_test.shape

def test_blog_so3():
    C = np.array(
        [
            SO3.Exp(np.random.normal(size=(3,)))
            for i in range(5)
        ]
    )
    phi = blog_so3(C)
    phi_test = np.array(
        [
            SO3.Log(C[i, :, :])
            for i in range(C.shape[0])
        ]
    ).squeeze()
    assert np.allclose(phi, phi_test)
    assert phi.shape == phi_test.shape

def test_blog_bexp_inverse():
    phi = np.random.normal(size=(5, 3))
    C = bexp_so3(phi)
    phi_test = blog_so3(C)
    assert np.allclose(phi, phi_test)
    assert phi.shape == phi_test.shape


if __name__ == "__main__":
    test_blog_bexp_inverse()