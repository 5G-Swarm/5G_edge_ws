import numpy as np
import math

class Pid_controller(object):
    def __init__(self,control_timestep,INTEGRAL_ERR_BOUND=1.5,OUTPUT_BOUND=2):

        self.control_timestep = control_timestep
        self.shape = 0
        self.pre_shape = 0
        # self.err_list = np.zeros(self.shape)
        # self.pre_err_list = np.zeros(self.shape)
        # self.integral_err_list = np.zeros(self.shape)
        # self.diff_err_list = np.zeros(self.shape)
        self.INTEGRAL_ERR_BOUND = INTEGRAL_ERR_BOUND
        self.OUTPUT_BOUND = OUTPUT_BOUND

    def control(self,ref_list,obs_list,kp = 1.0,ki = 0.0, kd = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err_list = ref_list - obs_list
        self.integral_err_list = self.integral_err_list + self.err_list * self.control_timestep
        self.integral_err_list = np.clip(self.integral_err_list, -self.INTEGRAL_ERR_BOUND, self.INTEGRAL_ERR_BOUND)
        self.diff_err_list = (self.err_list - self.pre_err_list)/self.control_timestep 
        #### PID target thrust #####################################
        # self.output_list = np.multiply(self.P_COEFF_FOR, self.err_list) \
        #                 + np.multiply(self.I_COEFF_FOR, self.integral_err_list) \
        #                 + np.multiply(self.D_COEFF_FOR, self.diff_err_list) 
        self.output_list = self.kp*self.err_list + self.ki*self.integral_err_list + self.kd*self.diff_err_list
        self.output_list = np.clip(self.output_list, -self.OUTPUT_BOUND, self.OUTPUT_BOUND)
        
        return self.output_list

    def err_control(self,err_list,kp = 1.0,ki = 0.0, kd = 0.0, OUTPUT_BOUND = 2):
        self.OUTPUT_BOUND = OUTPUT_BOUND
        self.shape = err_list.shape[0]
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err_list = err_list
        if self.pre_shape ==0:
            self.integral_err_list = np.zeros(self.shape) 
            self.pre_err_list = np.zeros(self.shape)           
        elif self.pre_shape < self.shape:
            self.integral_err_list = np.hstack((self.integral_err_list,np.zeros(self.shape-self.pre_shape)))
            self.pre_err_list = np.hstack((self.pre_err_list,np.zeros(self.shape-self.pre_shape)))
        self.integral_err_list = self.integral_err_list + self.err_list * self.control_timestep
        self.integral_err_list = np.clip(self.integral_err_list, -self.INTEGRAL_ERR_BOUND, self.INTEGRAL_ERR_BOUND)
        self.diff_err_list = (self.err_list - self.pre_err_list)/self.control_timestep 
        #### PID target thrust #####################################
        # self.output_list = np.multiply(self.P_COEFF_FOR, self.err_list) \
        #                 + np.multiply(self.I_COEFF_FOR, self.integral_err_list) \
        #                 + np.multiply(self.D_COEFF_FOR, self.diff_err_list) 
        self.output_list = self.kp*self.err_list + self.ki*self.integral_err_list + self.kd*self.diff_err_list
        self.output_list = np.clip(self.output_list, -self.OUTPUT_BOUND, self.OUTPUT_BOUND)
        
        return self.output_list

class Visual_servo(object):
    def __init__(self,control_timestep,OUTPUT_BOUND=0.2):
        self.controller = Pid_controller(control_timestep,OUTPUT_BOUND=OUTPUT_BOUND)


    def control(self,obswz_dict,obslocal_vz_dict,kp = 1.0,ki = 0.0, kd = 0.0, OUTPUT_BOUND=0.25):
        self.output_wz_list = self.controller.err_control(obswz_dict,kp = kp,ki = ki,kd = kd,OUTPUT_BOUND = OUTPUT_BOUND)
        self.output_lvz_list = self.controller.err_control(obslocal_vz_dict,kp = kp,ki = ki,kd = kd,OUTPUT_BOUND = OUTPUT_BOUND*0.4)*0.5

        return self.output_wz_list,self.output_lvz_list


