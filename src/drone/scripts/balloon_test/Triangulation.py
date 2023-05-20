#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def triangulation(pointl_vec,pointr_vec,R,t,cam_matrix_left,cam_matrix_right,n):
    pointl_cam_vec = []
    pointr_cam_vec = []
    for pointl in pointl_vec:
        # import pdb;pdb.set_trace()
        pointl_cam_vec.append([(pointl[0] - cam_matrix_left[0, 2]) / cam_matrix_left[0, 0],(pointl[1] - cam_matrix_left[1, 2]) / cam_matrix_left[1, 1]])
    for pointr in pointr_vec:
        pointr_cam_vec.append([(pointr[0] - cam_matrix_right[0, 2]) / cam_matrix_right[0, 0],(pointr[1] - cam_matrix_right[1, 2]) / cam_matrix_right[1, 1]])
    T1 = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]])
    T2 = np.concatenate((R,t),axis=1)
    # print(T1,T2)
    pointl_cam_vec = np.array(pointl_cam_vec).transpose()
    pointr_cam_vec = np.array(pointr_cam_vec).transpose()
    # print(pointl_cam_vec)
    # print(pointr_cam_vec)

    pts_4d = np.zeros((4,n))
    cv2.triangulatePoints(T1,T2,pointl_cam_vec,pointr_cam_vec,pts_4d)
    pts_3d = []
    for i in range(n):
        x = pts_4d[0,i]/pts_4d[3,i]/1000
        y = pts_4d[1,i]/pts_4d[3,i]/1000
        z = pts_4d[2,i]/pts_4d[3,i]/1000
        
        pts_3d.append([x,y,z])
    # pts_3d = np.array(pts_3d)
    print(pts_3d)
    return pts_3d





# class USB_Camera(object): 
#     def __init__(self):
#         # 左相机内参 及 畸变系数
#         self.cam_matrix_left = np.array([[1271.71,-0.1385,376.5282],[0.,1270.87,258.1373],[0.,0.,1.]])
#         self.distortion_l = np.array([[-0.5688,5.9214,-0.00038018,-0.00052731,-61.7538]])
#         # 右相机内参 及 畸变系数
#         self.cam_matrix_right = np.array([[1269.2524,-2.098026,367.9874],[0.,1267.1973,246.2712],[0.,0.,1.]])
#         self.distortion_r = np.array([[-0.5176,2.4704,-0.0011,0.0012,-16.1715]])
#         # 右边相机相对于左边相机的 旋转矩阵 R ， 平移矩阵 T
#         self.R = np.array([[1.0000,-0.0088,-0.0043],[0.0088,0.9999,0.0072],[0.0042,-0.0072,1.0000]])
#         self.T = np.array([[-68.5321],[-0.5832],[-4.0933]])
#         # 焦距
#         self.focal_length = 6.00
#         # 左右相机之间的距离 取 T 向量的第一维数值 单位 mm
#         self.baseline = np.abs(self.T[0])

class Drone_camera(object):
    def __init__(self,gps_pos,imu_ori,extrinsic_mat,intrinsic_mat,distortion):
        self.transform_world = np.eyes(4)
        self.transform_world[:3,:3] = Rotation.from_quat([imu_ori.x,imu_ori.y,imu_ori.z,imu_ori.w]).as_matrix()
        self.transform_world[:3,3] = np.array([gps_pos.longtitude,gps_pos.latitude,gps_pos.altitude])
        self.extrinsic_mat = extrinsic_mat
        self.intrinsic_mat = intrinsic_mat
        self.distortion = distortion
        self.pix_detect = np.array([0,0]) 
    
    def update(self,gps_pos,imu_ori,pix_detect):
        self.transform_world[:3,:3] = Rotation.from_quat([imu_ori.x,imu_ori.y,imu_ori.z,imu_ori.w]).as_matrix()
        self.transform_world[:3,3] = np.array([gps_pos.longtitude,gps_pos.latitude,gps_pos.altitude])
        self.pix_detect = np.array(pix_detect) 


class USB_Camera(object): 
    def __init__(self,left_drone,right_drone):
        # 左相机内参 及 畸变系数
        self.cam_matrix_left = left_drone.intrinsic_mat#np.array([[1271.71,-0.1385,376.5282],[0.,1270.87,258.1373],[0.,0.,1.]])
        self.distortion_l = left_drone.distortion#np.array([[-0.5688,5.9214,-0.00038018,-0.00052731,-61.7538]])
        # 右相机内参 及 畸变系数
        self.cam_matrix_right = right_drone.intrinsic_mat#np.array([[1269.2524,-2.098026,367.9874],[0.,1267.1973,246.2712],[0.,0.,1.]])
        self.distortion_r = right_drone.distortion#np.array([[-0.5176,2.4704,-0.0011,0.0012,-16.1715]])
        # 右边相机相对于左边相机的 旋转矩阵 R ， 平移矩阵 T
        self.tran_l_to_r = np.multiply(np.linalg.inv(left_drone.transform_world),right_drone.transform_world)
        self.R = self.tran_l_to_r[:3,:3]#np.array([[1.0000,-0.0088,-0.0043],[0.0088,0.9999,0.0072],[0.0042,-0.0072,1.0000]])
        self.T = self.tran_l_to_r[:3,3].reshape(3,1)#np.array([[-68.5321],[-0.5832],[-4.0933]])
        # 焦距
        self.focal_length = 6.00
        # 左右相机之间的距离 取 T 向量的第一维数值 单位 mm
        self.baseline = np.abs(self.T[0])
        self.Roi_points_left = self.left_drone.pix_detect
        self.Roi_points_right = self.right_drone.pix_detect


def test():
    left_drone = Drone_camera()
    right_drone = Drone_camera()
    config = USB_Camera(left_drone,right_drone)
    Roi_points_right = [
    # [164.0, 634.0],
    # [257.0, 602.0],
    # [351.0, 524.0],
    # [413.0, 446.0],
    # [460.0, 383.0],
    # [117.0, 430.0],
    # [117.0, 336.0],
    # [117.0, 273.0],
    # [117.0, 211.0],
    # [179.0, 414.0],
    # [194.0, 273.0],
    # [210.0, 195.0],
    # [226.0, 148.0],
    # [241.0, 399.0],
    # [272.0, 258.0],
    # [289.0, 180.0],
    # [319.0, 133.0],
    # [304.0, 414.0],
    # [350.0, 304.0],
    # [367.0, 242.0],
    [397.0, 180.0]
    ]
    Roi_points_left = [
    # [351.0, 649.0],
    # [445.0, 618.0],
    # [538.0, 555.0],
    # [600.0, 477.0],
    # [647.0, 399.0],
    # [304.0, 446.0],
    # [288.0, 351.0],
    # [288.0, 289.0],
    # [289.0, 242.0],
    # [366.0, 430.0],
    # [366.0, 289.0],
    # [382.0, 226.0],
    # [398.0, 164.0],
    # [414.0, 414.0],
    # [444.0, 273.0],
    # [460.0, 211.0],
    # [476.0, 148.0],
    # [476.0, 430.0],
    # [523.0, 320.0],
    # [553.0, 258.0],
    [569.0, 211.0]
    ]
    pts_3d = triangulation(config.Roi_points_left,config.Roi_points_right,config.R,config.T,config.cam_matrix_left,config.cam_matrix_right,1)
    pts_3d_norm=np.linalg.norm(pts_3d, axis=1, keepdims=True)

    for i in range(1):
        dist = pts_3d_norm[i][0]/1000
        print("distance of point {} is {:.4f}m".format(i,dist))
        print(pts_3d[i])


# test()