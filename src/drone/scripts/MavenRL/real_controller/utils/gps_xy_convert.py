import math
import numpy as np
CONSTANTS_RADIUS_OF_EARTH = 6371000.     # meters (m)
import utm



def gps2xy_new(lat,lon):
    return utm.from_latlon(lat,lon)[:2]

def xy2gps_new(x,y):
    return utm.to_latlon(x,y,51,"R")

gps_center = np.array([120.1196572,30.2658274])
gps_wn_en_es_ws = np.array([[120.1195138,30.2663404],[120.1201523,30.2661264],[120.1198082,30.2652929],[120.1191583,30.265505]])


def gps2xy(longtitude, latitude):
    L = 6381372*math.pi*2
    W = L
    H = L/2
    mill = 2.3
    x = longtitude*math.pi/180
    y = latitude*math.pi/180
    y = 1.25*math.log(math.tan(0.25*math.pi+0.4*y))
    x = (W/2)+(W/(2*math.pi))*x
    y = (H/2)-(H/(2*mill))*y
    return x, y

def xy2gps(x, y):
    L = 6381372 * math.pi*2
    W = L
    H = L/2
    mill = 2.3
    latitude = ((H/2-y)*2*mill)/(1.25*H)
    latitude = ((math.atan(math.exp(latitude))-0.25*math.pi)*180)/(0.4*math.pi)
    longtitude = (x-W/2)*360/W
    return round(latitude,7), round(longtitude,7)

def shahao_xy2gps():
    origin_gps = np.array([30.2658274,120.1196572])
    origin_big = np.array(gps2xy_new(origin_gps[0],origin_gps[1]))
    xy_list = np.array([[1.81418015761, 43.2812501076],
                        [-1.685819, 43.281],#balloon init
                        [-0.38917460810625926, 62.509860570542514],#task1 start_p
                        [-5.393152384960558, -60.75439882557839],#task1 end_p
                        [2.60645334394,42.9018380232],#task2 init
                        [-12.0,55],#set_1
                        [10,-19],#set_2
                        [11,-58]])#set_3
    gps_list = []
    for i in range(xy_list.shape[0]):
        temp_big = origin_big + xy_list[i]
        gps_list.append(list(xy2gps_new(temp_big[0],temp_big[1]))) 
    print(gps_list)

# shahao_xy2gps()

gps_first =  np.array([30.2618204,120.1172463])#wei,jing 120.1172463 30.2618204

gps_second =  np.array([30.2613856,120.1170687])#gps_imu 120.1170687 30.2613856

gps_first =  np.array([30.2612828,120.1169667])#gps_imu 120.1169667 30.2612828

gps_second =  np.array([30.2612417,120.1170945])#gps_imu 120.1170945 30.2612417

gps_first =  np.array([30.2618038,120.1172419])#gps_imu 120.1169667 30.2612828

gps_second =  np.array([30.2613618,120.1170659])#gps_imu 120.1170945 30.2612417

#temp
# 1. 120.1172419 30.2618038
# 2. 120.1170659 30.2613618



xy_first = np.array(gps2xy_new(gps_first[0],gps_first[1]))
xy_second = np.array(gps2xy_new(gps_second[0],gps_second[1]))

print("distance:",np.linalg.norm(xy_first-xy_second))


# gps_start = [30.2661264,120.1201523]
# gps_stop = [30.265505,120.1191583]#[120.119662, 30.26526] 

# xy_start = gps2xy_new(gps_start[0],gps_start[1])
# xy_stop = gps2xy_new(gps_stop[0],gps_stop[1])
# print(type(xy_stop),xy_stop) 
# # xy_start = [ -7.97325472315,47.0149079102]
# # xy_stop = [ 1.3408122461,-51.6518957034]
# # xy_stop = [ ]
# dis = np.linalg.norm(np.array(xy_start)-np.array(xy_stop))
# print("dis",dis)
# def GPStoXY(lat, lon, ref_lat, ref_lon):
#         # input GPS and Reference GPS in degrees
#         # output XY in meters (m) X:North Y:East
#         lat_rad = math.radians(lat)
#         lon_rad = math.radians(lon)
#         ref_lat_rad = math.radians(ref_lat)
#         ref_lon_rad = math.radians(ref_lon)

#         sin_lat = math.sin(lat_rad)
#         cos_lat = math.cos(lat_rad)
#         ref_sin_lat = math.sin(ref_lat_rad)
#         ref_cos_lat = math.cos(ref_lat_rad)

#         cos_d_lon = math.cos(lon_rad - ref_lon_rad)

#         arg = np.clip(ref_sin_lat * sin_lat + ref_cos_lat * cos_lat * cos_d_lon, -1.0, 1.0)
#         c = math.acos(arg)

#         k = 1.0
#         if abs(c) > 0:
#             k = (c / math.sin(c))

#         x = float(k * (ref_cos_lat * sin_lat - ref_sin_lat * cos_lat * cos_d_lon) * CONSTANTS_RADIUS_OF_EARTH)
#         y = float(k * cos_lat * math.sin(lon_rad - ref_lon_rad) * CONSTANTS_RADIUS_OF_EARTH)

#         return x, y


# def XYtoGPS(x, y, ref_lat, ref_lon):
#         x_rad = float(x) / CONSTANTS_RADIUS_OF_EARTH
#         y_rad = float(y) / CONSTANTS_RADIUS_OF_EARTH
#         c = math.sqrt(x_rad * x_rad + y_rad * y_rad)

#         ref_lat_rad = math.radians(ref_lat)
#         ref_lon_rad = math.radians(ref_lon)

#         ref_sin_lat = math.sin(ref_lat_rad)
#         ref_cos_lat = math.cos(ref_lat_rad)

#         if abs(c) > 0:
#             sin_c = math.sin(c)
#             cos_c = math.cos(c)

#             lat_rad = math.asin(cos_c * ref_sin_lat + (x_rad * sin_c * ref_cos_lat) / c)
#             lon_rad = (ref_lon_rad + math.atan2(y_rad * sin_c, c * ref_cos_lat * cos_c - x_rad * ref_sin_lat * sin_c))

#             lat = math.degrees(lat_rad)
#             lon = math.degrees(lon_rad)

#         else:
#             lat = math.degrees(ref_lat)
#             lon = math.degrees(ref_lon)

#         return lat, lon
