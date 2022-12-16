# %%
from pymocap import MocapTrajectory, MagnetometerData
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import rosbag
import numpy as np
from pynav.utils import set_axes_equal
sns.set_theme(style="whitegrid")
filename = "data/imu_calib.bag"
agent = "ifo003"

# Extract data
with rosbag.Bag(filename) as bag:
    mocap = MocapTrajectory.from_bag(bag, agent)
    mag = MagnetometerData.from_bag(bag, f"/{agent}/mavros/imu/mag")

D, b, m_a = mag.calibrate(mocap)
mag_calib = mag.apply_calibration(D, b)
# %% Plotting
mag.plot(mocap, mag_vector=m_a)
m = mag.magnetic_field
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(m[:,0],m[:,1],m[:,2])
set_axes_equal(ax)
ax.set_box_aspect([1,1,1])

fig, ax = plt.subplots(1,1)
ax.plot(mag.stamps, np.linalg.norm(mag.magnetic_field, axis=1))
ax.set_ylim(0,None)



mag_calib.plot(mocap, mag_vector=m_a)
fig, ax = plt.subplots(1,1)
m = mag_calib.magnetic_field
ax.scatter(m[:,0],m[:,1])
ax.scatter(m[:,1],m[:,2])
ax.scatter(m[:,0],m[:,2])
ax.axis("equal")
fig, ax = plt.subplots(1,1)
ax.plot(mag_calib.stamps, np.linalg.norm(mag_calib.magnetic_field, axis=1))
ax.set_ylim(0,None)


# def fitEllipsoid(magX, magY, magZ):
#     a1 = magX ** 2
#     a2 = magY ** 2
#     a3 = magZ ** 2
#     a4 = 2 * np.multiply(magY, magZ)
#     a5 = 2 * np.multiply(magX, magZ)
#     a6 = 2 * np.multiply(magX, magY)
#     a7 = 2 * magX
#     a8 = 2 * magY
#     a9 = 2 * magZ
#     a10 = np.ones(len(magX)).T
#     D = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])

#     # Eqn 7, k = 4
#     C1 = np.array([[-1, 1, 1, 0, 0, 0],
#                    [1, -1, 1, 0, 0, 0],
#                    [1, 1, -1, 0, 0, 0],
#                    [0, 0, 0, -4, 0, 0],
#                    [0, 0, 0, 0, -4, 0],
#                    [0, 0, 0, 0, 0, -4]])

#     # Eqn 11
#     S = np.matmul(D, D.T)
#     S11 = S[:6, :6]
#     S12 = S[:6, 6:]
#     S21 = S[6:, :6]
#     S22 = S[6:, 6:]

#     # Eqn 15, find eigenvalue and vector
#     # Since S is symmetric, S12.T = S21
#     tmp = np.matmul(np.linalg.inv(C1), S11 - np.matmul(S12, np.matmul(np.linalg.inv(S22), S21)))
#     eigenValue, eigenVector = np.linalg.eig(tmp)
#     u1 = eigenVector[:, np.argmax(eigenValue)]

#     # Eqn 13 solution
#     u2 = np.matmul(-np.matmul(np.linalg.inv(S22), S21), u1)

#     # Total solution
#     u = np.concatenate([u1, u2]).T

#     Q = np.array([[u[0], u[5], u[4]],
#                   [u[5], u[1], u[3]],
#                   [u[4], u[3], u[2]]])

#     n = np.array([[u[6]],
#                   [u[7]],
#                   [u[8]]])

#     d = u[9]

#     return Q, n, d


# import scipy
# magX = mag.magnetic_field[:,0]
# magY = mag.magnetic_field[:,1]
# magZ = mag.magnetic_field[:,2]

# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(111, projection='3d')

# ax1.scatter(magX, magY, magZ, s=5, color='r')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')

# # plot unit sphere
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = np.outer(np.cos(u), np.sin(v))
# y = np.outer(np.sin(u), np.sin(v))
# z = np.outer(np.ones(np.size(u)), np.cos(v))
# ax1.plot_wireframe(x, y, z, rstride=10, cstride=10, alpha=0.5)
# ax1.plot_surface(x, y, z, alpha=0.3, color='b')

# Q, n, d = fitEllipsoid(magX, magY, magZ)

# Qinv = np.linalg.inv(Q)
# b = -np.dot(Qinv, n)
# Ainv = np.real(1 / np.sqrt(np.dot(n.T, np.dot(Qinv, n)) - d) * scipy.linalg.sqrtm(Q))

# print("A_inv: ")
# print(Ainv)
# print()
# print("b")
# print(b)
# print()

# calibratedX = np.zeros(magX.shape)
# calibratedY = np.zeros(magY.shape)
# calibratedZ = np.zeros(magZ.shape)

# totalError = 0
# for i in range(len(magX)):
#     h = np.array([[magX[i], magY[i], magZ[i]]]).T
#     hHat = np.matmul(Ainv, h-b)
#     calibratedX[i] = hHat[0]
#     calibratedY[i] = hHat[1]
#     calibratedZ[i] = hHat[2]
#     mag = np.dot(hHat.T, hHat)
#     err = (mag[0][0] - 1)**2
#     totalError += err
# print("Total Error: %f" % totalError)

# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111, projection='3d')

# ax2.scatter(calibratedX, calibratedY, calibratedZ, s=1, color='r')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')

# # plot unit sphere
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = np.outer(np.cos(u), np.sin(v))
# y = np.outer(np.sin(u), np.sin(v))
# z = np.outer(np.ones(np.size(u)), np.cos(v))
# ax2.plot_wireframe(x, y, z, rstride=10, cstride=10, alpha=0.5)
# ax2.plot_surface(x, y, z, alpha=0.3, color='b')
# plt.show()




plt.show()