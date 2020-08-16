#!/usr/bin/env python
"""
Performs eye in hand (eih) image based visual servoing (ibvs).
Written by Alex Zhu (alexzhu(at)seas.upenn.edu)
"""
import numpy as np
import cv2
import sys
import time
import consts


class IbvsEih(object):
    """
    Performs eye in hand (eih) image based visual servoing (ibvs).
    """

    def __init__(self):
        self._final_cam_depth = None
        self._ideal_corners = None
        self._target_set = False
        self._L = np.zeros((2 * 4, 6))
        self._ideal_feature = np.zeros((4 * 2, 1))
        self._lambda = 0.5

        self.error = 100.0
        self.img = None
        self.num = 0

    def new_image_arrived(self):
        # TODO: new imageがあるかどうかチェックする
        img = cv2.imread("./ffc3_ibvs_{}.JPG".format(self.num % 4))
        self.num += 1
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return True

    def _initialize_jacobian(self):
        """
        画像ヤコビアンの近似
        :return:
        """
        for i in range(0, 4):
            x = self._ideal_corners[i * 2]
            y = self._ideal_corners[i * 2 + 1]
            self._ideal_feature[i * 2, 0] = x
            self._ideal_feature[i * 2 + 1, 0] = y
            Z = self._final_cam_depth
            self._L[i * 2:i * 2 + 2, :] = np.matrix(
                [[-1 / Z, 0, x / Z, x * y, -(1 + x * x), y], [0, -1 / Z, y / Z, 1 + y * y, -x * y, -x]])

    def _get_detected_corners(self):
        feature = None
        if self.img is not None:
            img = self.img
        mask = []
        hsv_min = np.array([consts.H_MIN, consts.S_MIN, consts.V_MIN])
        hsv_max = np.array([consts.H_MAX, consts.S_MAX, consts.V_MAX])
        mask_im = cv2.inRange(img, hsv_min, hsv_max)
        if np.sum(mask_im) > consts.AREA_MIN_EIH:
            mask.append(mask_im)

        # それっぽいのが見つかったものだけ保存
        if len(mask) == 0:
            print("NO endpoint found")
            sys.exit(1)
        elif len(mask) > 1:
            print("Multiple endpoints found")
        mask = mask[0]

        # masked_img = cv2.bitwise_and(img, img, mask=mask)

        # 輪郭抽出
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []
        for i, contour in enumerate(contours):
            # 小さすぎる/大きすぎる領域は除く
            area = cv2.contourArea(contour)
            if consts.AREA_MIN_EIH < area:
                final_contours.append(contour)
        # TODO: 複数見つかった対策
        if len(final_contours) == 1:
            final_contour = final_contours[0]
            # 矩形
            rect = cv2.minAreaRect(final_contour)
            box = cv2.boxPoints(rect)
            feature = np.int0(box)
            if np.linalg.norm(feature[2]-feature[3]) > np.linalg.norm(feature[2]-feature[1]):
                feature = feature[[2,3,0,1], :]
            else:
                feature = feature[[1,2,3,0], :]
        else:
            pass

        return feature

    def _command_velocity(self, vel):
        # TODO : MXT command
        print(vel)

    def get_next_vel(self, corners=None):

        if corners is None:
            return
        target_feature = corners.flatten()
        target_feature = target_feature[:, None]
        L = self._L

        self.error = target_feature - self._ideal_feature

        vel = -self._lambda * np.dot(np.linalg.pinv(L), self.error)

        return vel

    def set_target(self, final_camera_depth, desired_corners):
        # 最終目的位置の設定、画像ヤコビアン近似
        self._final_cam_depth = final_camera_depth
        self._ideal_corners = desired_corners
        self._initialize_jacobian()
        self._target_set = True

    def move_to_position(self, final_camera_depth, desired_corners, dist_tol):
        """
        Runs one instance of the visual servoing control law. Call when a new
        image has arrived.
        """
        self.set_target(final_camera_depth, desired_corners)

        while np.linalg.norm(self.error) > dist_tol:
            if not self.new_image_arrived():
                continue

            # Continue if no corners detected
            corners = self._get_detected_corners()
            if corners is None:
                continue

            # Don't move if the target hasn't been set
            if not self._target_set:
                continue

            # Get control law velocity and transform to body frame, then send to robot
            servo_vel = self.get_next_vel(corners=corners)
            self._command_velocity(servo_vel)

            time.sleep(1)



def main():
    # Set desired camera depth and desired feature coordinates as well as distance from goal before stopping
    final_camera_depth = 50
    desired_corners = np.array([60, 40, 90, 40, 60, 52, 90, 52])
    dist_tol = 2

    ibvseih = IbvsEih()
    ibvseih.move_to_position(final_camera_depth, desired_corners, dist_tol)


if __name__ == "__main__":
    main()

