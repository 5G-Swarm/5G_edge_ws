#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from time import sleep
import numpy as np
import threading

from informer import Informer

# robot_num = 10
ROBOT_LIST = [18, 19]
ifm_r2e_dict = {}
ifm_e2c_dict = {}

def parse_recg(message, robot_id):
    # print('parse_recg', len(message), robot_id)
    relay_recg(message, robot_id)

def relay_recg(message, robot_id):
    global ifm_e2c_dict
    if robot_id in ifm_e2c_dict.keys():
        # print('send recg to', robot_id)
        ifm_e2c_dict[robot_id].send_recg(message)

def parse_img(message, robot_id):
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
    #print('get odm', robot_id)
    relay_odm(message, robot_id)

def relay_odm(message, robot_id):
    global ifm_e2c_dict
    if robot_id in ifm_e2c_dict.keys():
        ifm_e2c_dict[robot_id].send_odm(message)

def parse_cmd(message, robot_id):
    relay_cmd(message, robot_id)

def relay_cmd(message, robot_id):
    global ifm_e2c_dict
    if robot_id in ifm_e2c_dict.keys():
        ifm_e2c_dict[robot_id].send_cmd(message)

class ServerR2E(Informer):
    def img_recv(self):
        self.recv('img', parse_img)

    def msg_recv(self):
        self.recv('msg', parse_msg)

    def odm_recv(self):
        self.recv('odm', parse_odm)

    def cmd_recv(self):
        self.recv('cmd', parse_cmd)

    def recg_recv(self):
        self.recv('recg', parse_recg)

    def send_path(self, message):
        self.send(message, 'path')

    def send_ctrl(self, message):
        self.send(message, 'ctrl')

#############################################

def parse_path(message, robot_id):
    relay_path(message, robot_id)

def relay_path(message, robot_id):
    global ifm_r2e_dict
    if robot_id in ifm_r2e_dict.keys():
        ifm_r2e_dict[robot_id].send_path(message)

def parse_ctrl(message, robot_id):
    relay_ctrl(message, robot_id)

def relay_ctrl(message, robot_id):
    global ifm_r2e_dict
    if robot_id in  ifm_r2e_dict.keys():
        ifm_r2e_dict[robot_id].send_ctrl(message)
class ServerE2C(Informer):
    def send_msg(self, message):
        self.send(message, 'msg')
    
    def send_odm(self, message):
        self.send(message, 'odm')

    def send_cmd(self, message):
        self.send(message, 'cmd')

    def send_img(self, message):
        self.send(message, 'img')

    def send_recg(self, message):
        self.send(message, 'recg')
    
    def path_recv(self):
        try:
            self.recv('path', parse_path)
        except:
            print('recv path timeout !')

    def ctrl_recv(self):
        try:
            self.recv('ctrl', parse_ctrl)
        except:
            print('recv ctrl timeout !')

def start_r2e():
    global ifm_r2e_dict
    for i in ROBOT_LIST:
        ifm_r2e_dict[i] = ServerR2E(config = 'config.yaml', robot_id = i)
    # ifm_r2e_dict[1] = ServerR2E(config = 'config.yaml', robot_id = 1)

def start_e2c():
    global ifm_e2c_dict
    for i in ROBOT_LIST:
        ifm_e2c_dict[i] = ServerE2C(config = 'config_e2c.yaml', robot_id = i)

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
