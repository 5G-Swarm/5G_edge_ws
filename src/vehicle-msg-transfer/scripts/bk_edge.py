#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from time import sleep
import numpy as np
import threading

from informer import Informer

robot_num = 1
ifm_r2e_dict = {}
ifm_e2c_dict = {}

def parse_img(message, robot_id):
    print('get img !!!', len(message), robot_id)
    relay_img(message, robot_id)
    # nparr = np.frombuffer(message, np.uint8)
    # img = cv2.imdecode(nparr,  cv2.IMREAD_COLOR)
    # cv2.imshow('Image',img)
    # cv2.waitKey(1)

def relay_img(message, robot_id):
    global ifm_e2c_dict
    if robot_id in ifm_e2c_dict.keys():
        ifm_e2c_dict[robot_id].send_img(message)

def parse_msg(message, robot_id):
    relay_msg(message, robot_id)

def relay_msg(message, robot_id):
    global ifm_e2c_dict
    if robot_id in ifm_e2c_dict.keys():
        ifm_e2c_dict[robot_id].send_msg(message)

def parse_odm(message, robot_id):
    relay_odm(message, robot_id)

def relay_odm(message, robot_id):
    global ifm_e2c_dict
    if robot_id in ifm_e2c_dict.keys():
        ifm_e2c_dict[robot_id].send_odm(message)

def parse_pcd(message, robot_id):
    #print('relay pcd')
    relay_pcd(message, robot_id)

def relay_pcd(message, robot_id):
    global ifm_e2c_dict
    if robot_id in ifm_e2c_dict.keys():
        ifm_e2c_dict[robot_id].send_pcd(message)

class ServerR2E(Informer):
    def img_recv(self):
        self.recv('img', parse_img)

    def msg_recv(self):
        self.recv('msg', parse_msg)

    def odm_recv(self):
        self.recv('odm', parse_odm)

    def pcd_recv(self):
        self.recv('pcd', parse_pcd)

class ServerE2C(Informer):
    def send_msg(self, message):
        self.send(message, 'msg')
    
    def send_odm(self, message):
        self.send(message, 'odm')

    def send_pcd(self, message):
        self.send(message, 'pcd')

    def send_img(self, message):
        self.send(message, 'img')

def start_r2e():
    global ifm_r2e_dict
    for i in range(0, robot_num):
        ifm_r2e_dict[i] = ServerR2E(config = 'config-edge-e2v.yaml', robot_id = i)

def start_e2c():
    global ifm_e2c_dict
    for i in range(0, robot_num):
        ifm_e2c_dict[i] = ServerE2C(config = 'config-edge-e2c.yaml', robot_id = i)

if __name__ == '__main__':
    start_r2e_thread = threading.Thread(
        target = start_r2e, args=()
    )
    start_e2c_thread = threading.Thread(
        target = start_e2c, args=()
    )
    start_r2e_thread.start()
    start_e2c_thread.start()
    while True:
        sleep(0.01)
