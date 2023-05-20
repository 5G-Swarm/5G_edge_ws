import numpy as np
temp_4g_4a_drone = np.array([14.988,14.606,14.606,14.989])
temp_4g_4a_edge = np.array([5.548,14.982,8.296,10.081])
print("temp_4g_4a_edge",np.sum(temp_4g_4a_edge)/np.sum(temp_4g_4a_drone))

temp_4g_3a_drone = np.array([14.605,15.012,14.989])
temp_4g_3a_edge = np.array([10.62,14.604,11.608])
print("temp_4g_3a_edge",np.sum(temp_4g_3a_edge)/np.sum(temp_4g_3a_drone))

temp_4g_2a_drone = np.array([14.605,14.993])
temp_4g_2a_edge = np.array([14.605,11.913])
print("temp_4g_2a_edge",np.sum(temp_4g_2a_edge)/np.sum(temp_4g_2a_drone))

temp_5g_7a_drone = np.array([14.605,14.987,14.990,14.989,14.985,14.986,14.984])
temp_5g_7a_edge = np.array([13.179,14.927,14.918,14.924,15.011,14.013,15.013])
print("temp_5g_7a_edge",np.sum(temp_5g_7a_edge)/np.sum(temp_5g_7a_drone))

temp_5g_6a_drone = np.array([14.985,14.988,14.989,14.988,14.605,14.154])
temp_5g_6a_edge = np.array([14.990,14.985,14.991,14.982,14.611,14.155])
print("temp_5g_6a_edge",np.sum(temp_5g_6a_edge)/np.sum(temp_5g_6a_drone))

temp_5g_5a_drone = np.array([14.986,14.988,14.989,14.988,14.605])
temp_5g_5a_edge = np.array([14.989,14.988,14.991,14.988,14.605])
print("temp_5g_5a_edge",np.sum(temp_5g_5a_edge)/np.sum(temp_5g_5a_drone))

temp_5g_4a_drone = np.array([14.986,14.988,14.989,14.988])
temp_5g_4a_edge = np.array([14.990,14.989,14.988,14.988])
print("temp_5g_4a_edge",np.sum(temp_5g_4a_edge)/np.sum(temp_5g_4a_drone))