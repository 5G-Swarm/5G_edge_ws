import numpy as np
from rtree import index

from itertools import tee

import numpy as np


def dist_between_points(a, b):
    """
    Return the Euclidean distance between two points
    :param a: first point
    :param b: second point
    :return: Euclidean distance between a and b
    """
    distance = np.linalg.norm(np.array(b) - np.array(a))
    return distance


def pairwise(iterable):
    """
    Pairwise iteration over iterable
    :param iterable: iterable
    :return: s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def es_points_along_line(start, end, r):
    """
    Equally-spaced points along a line defined by start, end, with resolution r
    :param start: starting point
    :param end: ending point
    :param r: maximum distance between points
    :return: yields points along line from start to end, separated by distance r
    """
    d = dist_between_points(start, end)
    n_points = int(np.ceil(d / r))
    if n_points > 1:
        step = d / (n_points - 1)
        for i in range(n_points):
            next_point = steer(start, end, i * step)
            yield next_point


def steer(start, goal, d):
    """
    Return a point in the direction of the goal, that is distance away from start
    :param start: start location
    :param goal: goal location
    :param d: distance away from start
    :return: point in the direction of the goal, distance away from start
    """
    start, end = np.array(start), np.array(goal)
    v = end - start
    u = v / (np.sqrt(np.sum(v ** 2)))
    steered_point = start + u * d
    return tuple(steered_point)


def relative_rt_formation_array_calculation(current_formation_array,affine_param_array):
    relative_theta = affine_param_array[0]
    relative_shear_x = affine_param_array[1]
    relative_scale_x = affine_param_array[2]
    relative_scale_y = affine_param_array[3]
    relative_transition_x = affine_param_array[4]
    relative_transition_y = affine_param_array[5]
    # import pdb;pdb.set_trace()
    relative_Rotation_matrix = np.array([[np.cos(relative_theta),-np.sin(relative_theta)],[np.sin(relative_theta),np.cos(relative_theta)]]) 
    relative_Shearing_matrix = np.array([[1,relative_shear_x],[0,1]])
    relative_Scaling_matrix = np.array([[np.exp(relative_scale_x),0],[0,np.exp(relative_scale_y)]])
    relative_A_matrix = relative_Rotation_matrix.dot(relative_Shearing_matrix.dot(relative_Scaling_matrix))
    relative_B_matrix = np.array([[relative_transition_x],[relative_transition_y]])

    # self.rt_affine_matrix = np.hstack((relative_A_matrix,np.zeros((2,1)))) 
    # print(current_formation_array,relative_A_matrix,relative_B_matrix)
    return relative_A_matrix.dot(current_formation_array) + relative_B_matrix 

class SearchSpace(object):
    def __init__(self, init_template, dimension_lengths,r, O=None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # sanity check
        self.init_template = init_template
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        p1 = index.Property()
        p1.dimension = 2#只在二维上产生障碍物self.dimensions
        p2 = index.Property()
        p2.dimension = 2
        if O is None:##
            self.obs = index.Index(interleaved=True, properties=p1)
            self.extend_obs =  index.Index(interleaved=True, properties=p2)
        else:
            # r-tree representation of obstacles
            # sanity check
            if any(len(o) / 2 != len(dimension_lengths) for o in O):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
                raise Exception("Obstacle start must be less than obstacle end")
            # self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)
        
        self.r = r

    def point_obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        return self.obs.count(x+x) == 0#self.extend_obs.count(x+x) == 0#


    def edge_obstacle_free(self, start, end):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        points = es_points_along_line(start, end, self.r)
        coll_free = all(map(self.point_obstacle_free, points))
        return coll_free



    def formation_obstacle_free(self, x):
        """
        Check if a formation resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        temp_formation_array = relative_rt_formation_array_calculation(self.init_template, x)
        for i in range(temp_formation_array.shape[1]):
            j = i+1 if i+1 < temp_formation_array.shape[1] else 0
            ###检查编队每一条边是否和障碍物碰撞
            if self.edge_obstacle_free(temp_formation_array[:,i],temp_formation_array[:,j]):
                continue

            return False

        
        return True#self.obs.count(x) == 0

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            if self.formation_obstacle_free(x):
                return x



    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """

        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        return tuple(x)
