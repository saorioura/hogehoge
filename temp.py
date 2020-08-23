import numpy as np
import cv2
from calib import draw, calibration
from get_ffc_end import get_ffc_end
import consts
import os
import math

input_img = cv2.imread("./ffc3_ibvs_0.JPG")

# カメラパラメータ
if os.path.isfile('./cameraparams.npz'):
    # 前に記憶しておいたデータを読みだす
    with np.load('./cameraparams.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
else:
    mtx, dist, _, _ = calibration()

# mask
hsv_min = np.array([90, 120, 100])
hsv_max = np.array([120, 255, 255])

hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_img, hsv_min, hsv_max)


masked_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)

# 輪郭抽出
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final_contours = []
for i, contour in enumerate(contours):
    # 小さすぎる/大きすぎる領域は除く
    area = cv2.contourArea(contour)
    print(area)
    if 200 < area:
        final_contours.append(contour)
final_contour = final_contours[0]


# 矩形
rect = cv2.minAreaRect(final_contour)
box = cv2.boxPoints(rect)
box = np.int0(box)

if np.linalg.norm([box[0]-box[1]]) > np.linalg.norm([box[0]-box[3]]) * 1.5:
    ffc_end_box = np.array([box[2], box[3], box[1], box[0]])
else:
    ffc_end_box = np.array([box[1], box[2], box[0], box[3]])

# PnP
objp = np.array([[0, 0, 0], [consts.FFC_W, 0, 0], [0, consts.FFC_H, 0], [consts.FFC_W, consts.FFC_H, 0]]).astype(np.float32)
corner = ffc_end_box.reshape([-1, 1, 2]).astype(np.float32)
ret, rvec, tvec = cv2.solvePnP(objp, corner, mtx, dist)



vector_0 = np.array([0, 0, 0]).reshape(-1, 1)
vector_1 = np.array([consts.FFC_W, 0, 0]).reshape(-1, 1)
vector_2 = np.array([consts.FFC_W, consts.FFC_H, 0]).reshape(-1, 1)
vector_3 = np.array([0, consts.FFC_H, 0]).reshape(-1, 1)
target_pos_0 = np.dot(cv2.Rodrigues(rvec)[0], vector_0) + tvec
target_pos_1 = np.dot(cv2.Rodrigues(rvec)[0], vector_1) + tvec
target_pos_2 = np.dot(cv2.Rodrigues(rvec)[0], vector_2) + tvec
target_pos_3 = np.dot(cv2.Rodrigues(rvec)[0], vector_3) + tvec
print("Target Z (Camera coordinate)")
print(target_pos_0[-1]/1000, target_pos_1[-1]/1000, target_pos_2[-1]/1000, target_pos_3[-1]/1000)


# ３次元の点を平面に投影
axis = np.float32([[consts.FFC_W,0,0], [0,consts.FFC_H,0], [0,0,-10]]).reshape(-1,3)
imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

img = draw(input_img, corner, imgpts)
cv2.imshow('img', input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
