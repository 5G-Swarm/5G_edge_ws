import numpy as np
from rtree import index

from itertools import tee

import numpy as np
from math import cos, sin, pi, sqrt


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
            self.extend_obs_for_opt =  index.Index(interleaved=True, properties=p2)
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
        return self.extend_obs.count(x+x) == 0##self.obs.count(x+x) == 0#self.extend_obs.count(x+x) == 0#


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


 
"""
Contains the class implementataion of the GJK Algorithm
"""

import numpy as np
# np.random.seed(0)




class GJK():
    def __init__(self, ObjectList=[], tolerance=0.00001):
        """
        Initialize the cobject environment
        """
        self.ObjCount = 0
        self.ObjID = {}
        self.ConvObject = []
        self.tolerance = tolerance
        for shape in ObjectList:
            assert len(shape) == 2, \
                "Shape name and a shape class object should be provided."
            assert type(shape[0]) == str, "Shape name must be a string"
            self.ObjID[shape[0]] = self.ObjCount
            self.ConvObject.append(shape[1])
            self.ObjCount += 1
        self.WitnessPts = {}

    def SupportMinkowskiDiff(self, Shape1, Shape2, Dir):
        """
        Calculate the support of the Minkowski difference Shape1-Shape2
        """
        Pt1, Dist1, Index1 = Shape1.SupportFunc(Dir)
        Pt2, Dist2, Index2 = Shape2.SupportFunc(-Dir)
        return np.array([Pt1-Pt2, Dist1+Dist2, (Index1, Index2)], dtype=object)

    def S1D(self, Simplex):
        """
        Sub-routine for 1-simplex.
        Searches the voronoi regions of a line.
        """
        if all(Simplex[1] == Simplex[0]):
            return Simplex[1:], np.ones(1, dtype='float64')
        t = Simplex[1]-Simplex[0]
        po = -(np.dot(Simplex[1], t)/np.dot(t, t))*t+Simplex[1]
        u_max = 0
        # Find on which axis to project the simplex to avoid degeneracy
        for i in range(3):
            u = Simplex[0][i]-Simplex[1][i]
            if abs(u) > abs(u_max):
                u_max = u
                Index = i
        # Only i-th co-ordinate is retained
        k = 1
        C2 = np.zeros(2, dtype='float64')
        # Check if the origin is on the line segment
        for j in range(2):
            C2[j] = ((-1)**(j+1))*(Simplex[k][Index]-po[Index])
            k = j
        if (u_max > 0 and all(C2 > 0)) or (u_max < 0 and all(C2 < 0)):
            return Simplex, C2/u_max
        else:
            # Find which end point of the line segment is closest to the origin
            if (np.linalg.norm(Simplex[0]) < np.linalg.norm(Simplex[1])):
                return Simplex[0:1], np.ones(1, dtype='float64')
            else:
                return Simplex[1:], np.ones(1, dtype='float64')

    def S2D(self, Simplex):
        """
        Sub-routine for 2-simplex.
        Searches the voronoi regions of a planar triangle.
        """
        n = np.cross(Simplex[1]-Simplex[0], Simplex[2]-Simplex[0])
        po = (np.dot(Simplex[0], n)/np.dot(n, n))*n

        # Find on which plane to project the simplex to avoid degeneracy
        u_max = 0
        k = 1
        l = 2
        for i in range(0, 3):
            u = ((-1)**(i))*(
                             Simplex[0][k]*Simplex[1][l]
                             + Simplex[1][k]*Simplex[2][l]
                             + Simplex[2][k]*Simplex[0][l]
                             - Simplex[1][k]*Simplex[0][l]
                             - Simplex[2][k]*Simplex[1][l]
                             - Simplex[0][k]*Simplex[2][l])
            if abs(u) > abs(u_max):
                u_max = u
                J = i
            k = l
            l = i
        # co-ordinate J is discarded
        x, y = np.delete(np.arange(0, 3), J)
        k = 1
        l = 2
        C3 = np.zeros(3, dtype='float64')
        # Check if the origin is within the triangle
        for j in range(3):
            C3[j] = (
                     po[x]*Simplex[k][y]+po[y]*Simplex[l][x]
                     + Simplex[k][x]*Simplex[l][y]
                     - po[x]*Simplex[l][y]-po[y]*Simplex[k][x]
                     - Simplex[l][x]*Simplex[k][y])
            k = l
            l = j
        if (u_max > 0 and all(C3 > 0)) or (u_max < 0 and all(C3 < 0)):
            return Simplex, C3/u_max
        d = np.Infinity
        # Find which side of the triangle is closest to the origin
        for j in range(0, 3):
            if (u_max >= 0 and -C3[j] >= 0) or (u_max <= 0 and -C3[j] <= 0):
                Simplex1D = np.delete(Simplex, j, axis=0)
                W_astrix, Lambda_astrix = self.S1D(Simplex1D)
                d_astrix = np.linalg.norm(np.matmul(Lambda_astrix, W_astrix))
                if d_astrix < d:
                    W = W_astrix
                    Lamda = Lambda_astrix
                    d = d_astrix
        return W, Lamda

    def S3D(self, Simplex):
        """
        Sub-routine for 3-simplex.
        Searches the voronoi regions of a tetrahedron.
        """
        M = np.vstack([Simplex.T, np.ones((1, 4), dtype='float64')])
        detM = 0
        C4 = np.zeros(4, dtype='float64')
        # Check if the origin is within the tetrahedron
        for j in range(4):
            C4[j] = ((-1)**((j+1)+4))*np.linalg.det(
                            np.hstack([M[0:3, 0:j], M[0:3, j+1:]]))
            detM += C4[j]
        if (detM > 0 and all(C4 > 0)) or (detM < 0 and all(C4 < 0)):
            return Simplex, C4/detM
        d = np.Infinity
        # Find which face of the tetrahedron is closest to the origin
        for j in range(0, 4):
            if (detM >= 0 and -C4[j] >= 0) or (detM <= 0 and -C4[j] <= 0):
                Simplex2D = np.delete(Simplex, j, axis=0)
                W_astrix, Lambda_astrix = self.S2D(Simplex2D)
                d_astrix = np.linalg.norm(np.matmul(Lambda_astrix, W_astrix))
                if d_astrix < d:
                    W = W_astrix
                    Lamda = Lambda_astrix
                    d = d_astrix
        return W, Lamda

    def SignedVolumes(self, Simplex):
        """
        Performs the signed volumes distance algorithm
        on the simplex specified
        """
        # Call routine based on simplex size
        if len(Simplex) == 4:
            return self.S3D(Simplex)
        elif len(Simplex) == 3:
            return self.S2D(Simplex)
        elif len(Simplex) == 2:
            return self.S1D(Simplex)
        elif len(Simplex) == 1:
            return Simplex, np.ones((1), dtype='float64')
        else:
            print("error")

    def GetDist(self, ShapeID1, ShapeID2):
        """
        Computes the distance between 2 convex bodies
        """
        k = 0
        Shape1 = self.ObjID[ShapeID1]
        Shape2 = self.ObjID[ShapeID2]
        # Initialize the simplex
        # Select a random initial point for the simplex
        index_1 = np.random.randint(0, self.ConvObject[Shape1].NoOfVertices)
        index_2 = np.random.randint(0, self.ConvObject[Shape2].NoOfVertices)
        InitialPt = (
                     self.ConvObject[Shape1].CurrentVertices[index_1] -
                     self.ConvObject[Shape2].CurrentVertices[index_2])

        Simplex = InitialPt.reshape((1, 3))
        # Set initial direction opposite to the selected point to maximize area
        NewPt, _, _ = self.SupportMinkowskiDiff(
                                                self.ConvObject[Shape1],
                                                self.ConvObject[Shape2],
                                                -InitialPt)

        Simplex = np.vstack([NewPt, Simplex])
        while True:
            k += 1
            # Use signed volumes to get the supporting points and their weights
            Simplex, Lambda = self.SignedVolumes(Simplex)
            vk = np.matmul(Lambda, Simplex)
            NewPt, hk, _ = self.SupportMinkowskiDiff(
                                                     self.ConvObject[Shape1],
                                                     self.ConvObject[Shape2],
                                                     -vk)
            vk_square = np.dot(vk, vk)
            gk = vk_square+hk
            if (gk < self.tolerance or len(Simplex) == 4):
                return np.sqrt(vk_square)
            Simplex = np.vstack([NewPt, Simplex])



class Polytope():
    def __init__(self, PointList):
        """
        Create and intialize a convex ploytope at the robot
        end effector
        """
        self.Vertices = np.array(PointList, dtype='float64')
        assert len(self.Vertices.shape) == 2, \
            "Atleast 1 point must be specified.\n" +\
            "All points must be in 3D space"
        assert self.Vertices.shape[1] == 3, "All points must be in 3D space"
        self.NoOfVertices = len(PointList)
        self.CurrentVertices = np.copy(self.Vertices)

    def UpdatePosition(self, pos, quat):
        """
        Updates the vertices of the polytope based on the
        provided position vector and quaternion
        """
        pos = np.array(pos, dtype="float64")
        quat = np.array(quat, dtype="float64")
        for i in range(len(self.Vertices)):
            self.CurrentVertices[i] = pos+RotatePos(self.Vertices[i], quat)

    def SupportFunc(self, dVector):
        """
        Provides the farthest point in the polytope along the
        provided direction dVector

        Inputs:
            dVector      - direction vector in 3D space
        Output:
            Farthest point in polytope along the direction dVector,
            Projection of the farthest point on the direction vector
        """
        MaxDist = np.dot(dVector, self.CurrentVertices[0])
        VertexIndex = 0
        for i in range(1, len(self.CurrentVertices)):
            dist = np.dot(dVector, self.CurrentVertices[i])
            if dist > MaxDist:
                MaxDist = dist
                VertexIndex = i
        return np.array([self.CurrentVertices[VertexIndex],
                         MaxDist, VertexIndex], dtype=object)

def Transf_Matrix(pos, quat):
    """
    Converts a position and quaternion value into
    a (4x4) Transformation matrix

    Arguments:
        pos - array (x, y, z)
        quat - array (q0, qx, qy, qz)
    """
    Result = np.zeros((4, 4))
    Result[3][3] = 1
    Result[0:3, 3] = pos
    Result[0:3, 0:3] = quat_to_homo(quat)
    return Result


def quat_to_homo(quat):
    """
    Converts a quaternion into a homogeneous matrix

    Arguments:
        quat - array (q0, qx, qy, qz)
    """
    ResultMatrix = np.ones((3, 3), dtype="float64")
    ResultMatrix[0, 0] = 1-2*(quat[2]**2+quat[3]**2)
    ResultMatrix[0, 1] = 2*(quat[1]*quat[2]-quat[3]*quat[0])
    ResultMatrix[0, 2] = 2*(quat[1]*quat[3]+quat[2]*quat[0])
    ResultMatrix[1, 0] = 2*(quat[1]*quat[2]+quat[3]*quat[0])
    ResultMatrix[1, 1] = 1-2*(quat[1]**2+quat[3]**2)
    ResultMatrix[1, 2] = 2*(quat[2]*quat[3]-quat[1]*quat[0])
    ResultMatrix[2, 0] = 2*(quat[1]*quat[3]-quat[2]*quat[0])
    ResultMatrix[2, 1] = 2*(quat[2]*quat[3]+quat[1]*quat[0])
    ResultMatrix[2, 2] = 1-2*(quat[1]**2+quat[2]**2)
    return ResultMatrix


def eulerzxz_to_quat(euler_z_x_z):
    """
    Converts a Euler ZXZ angles into quaternion values

    Arguments:
        euler_z_x_z - array (eu_z1, eu_x, eu_z2)
    """
    euz = euler_z_x_z[0]*pi/180
    eux_d = euler_z_x_z[1]*pi/180
    euz_dd = euler_z_x_z[2]*pi/180
    cos_euz = cos(euz)
    sin_euz = sin(euz)
    cos_eux_d = cos(eux_d)
    sin_eux_d = sin(eux_d)
    cos_euz_dd = cos(euz_dd)
    sin_euz_dd = sin(euz_dd)

    m_11 = cos_euz*cos_euz_dd-cos_eux_d*sin_euz*sin_euz_dd
    m_12 = cos_euz_dd*sin_euz+cos_eux_d*cos_euz*sin_euz_dd
    m_13 = sin_eux_d*sin_euz_dd
    m_21 = -cos_euz*sin_euz_dd-cos_eux_d*cos_euz_dd*sin_euz
    m_22 = -sin_euz*sin_euz_dd+cos_eux_d*cos_euz*cos_euz_dd
    m_23 = sin_eux_d*cos_euz_dd
    m_31 = sin_eux_d*sin_euz
    m_32 = -cos_euz*sin_eux_d
    m_33 = cos_eux_d

    rot = np.array([[m_11, m_12, m_13],
                    [m_21, m_22, m_23],
                    [m_31, m_32, m_33]], dtype='float64')

    rot = rot.transpose()

    if (rot[2][1] - rot[1][2]) < 0:
        q2m = -1
    else:
        q2m = 1

    if (rot[0][2] - rot[2][0]) < 0:
        q3m = -1
    else:
        q3m = 1

    if (rot[1][0] - rot[0][1]) < 0:
        q4m = -1
    else:
        q4m = 1

    q = np.zeros(4, dtype='float64')
    q[0] = sqrt(rot[0][0] + rot[1][1] + rot[2][2] + 1)/2
    q[1] = q2m * sqrt(rot[0][0] - rot[1][1] - rot[2][2] + 1)/2
    q[2] = q3m * sqrt(rot[1][1] - rot[0][0] - rot[2][2] + 1)/2
    q[3] = q4m * sqrt(rot[2][2] - rot[0][0] - rot[1][1] + 1)/2
    return q


def eulerzyx_to_quat(euler_z_y_x):
    """
    Converts a Euler ZYX angles into quaternion values

    Arguments:
        euler_z_y_x - array (eu_z, eu_y, eu_x)
    """
    eux = euler_z_y_x[2]*pi/180
    euy = euler_z_y_x[1]*pi/180
    euz = euler_z_y_x[0]*pi/180
    cos_eux = cos(eux)
    sin_eux = sin(eux)
    cos_euy = cos(euy)
    sin_euy = sin(euy)
    cos_euz = cos(euz)
    sin_euz = sin(euz)

    m_11 = cos_euy*cos_euz
    m_12 = sin_eux*sin_euy*cos_euz - sin_euz*cos_eux
    m_13 = sin_eux*sin_euz + sin_euy*cos_eux*cos_euz
    m_21 = sin_euz*cos_euy
    m_22 = sin_eux*sin_euy*sin_euz + cos_eux*cos_euz
    m_23 = -sin_eux*cos_euz + sin_euy*sin_euz*cos_eux
    m_31 = -sin_euy
    m_32 = sin_eux*cos_euy
    m_33 = cos_eux*cos_euy

    rot = np.array([[m_11, m_12, m_13],
                    [m_21, m_22, m_23],
                    [m_31, m_32, m_33]], dtype='float64')

    if (rot[2][1] - rot[1][2]) < 0:
        q2m = -1
    else:
        q2m = 1
    if (rot[0][2] - rot[2][0]) < 0:
        q3m = -1
    else:
        q3m = 1
    if (rot[1][0] - rot[0][1]) < 0:
        q4m = -1
    else:
        q4m = 1

    q = np.zeros(4, dtype='float64')
    q[0] = sqrt(rot[0][0] + rot[1][1] + rot[2][2] + 1)/2
    q[1] = q2m * sqrt(rot[0][0] - rot[1][1] - rot[2][2] + 1)/2
    q[2] = q3m * sqrt(rot[1][1] - rot[0][0] - rot[2][2] + 1)/2
    q[3] = q4m * sqrt(rot[2][2] - rot[0][0] - rot[1][1] + 1)/2
    return q


def rot_to_quat(rot):
    """
    Converts a rotation matrix into a quaternion

    Arguments:
        rot - array (3x3)
    """
    if (rot[2][1] - rot[1][2]) < 0:
        q2m = -1
    else:
        q2m = 1

    if (rot[0][2] - rot[2][0]) < 0:
        q3m = -1
    else:
        q3m = 1

    if (rot[1][0] - rot[0][1]) < 0:
        q4m = -1
    else:
        q4m = 1

    q = np.zeros(4, dtype='float64')

    q[0] = sqrt(rot[0][0] + rot[1][1] + rot[2][2] + 1)/2
    q[1] = q2m * sqrt(rot[0][0] - rot[1][1] - rot[2][2] + 1)/2
    q[2] = q3m * sqrt(rot[1][1] - rot[0][0] - rot[2][2] + 1)/2
    q[3] = q4m * sqrt(rot[2][2] - rot[0][0] - rot[1][1] + 1)/2
    return q


def QuatMult(quat1, quat2):
    """
    Multiplies two quaternion

    Arguments:
        quat1 - array (q0, qx, qy, qz)
        quat2 - array (q0, qx, qy, qz)
    """
    result = np.zeros(4, dtype='float64')
    result[0] = quat1[0]*quat2[0]-np.dot(quat1[1:], quat2[1:])
    result[1:] = quat1[0]*quat2[1:]+quat2[0]*quat1[1:] +\
        np.cross(quat1[1:], quat2[1:])
    return result


def RotatePos(pos, quat):
    """
    Rotates the provided pos by the specified quaternion

    Arguments:
        pos - array (x, y, z)
        quat - array (q0, qx, qy, qz)
    """
    assert len(pos) == 3, "Specify coordinates in 3D space"
    assert len(quat) == 4, "Provide unit quaternion values"
    assert abs(np.linalg.norm(quat)-1) < 0.00001, \
        "Quaternion does not have unit norm"
    temp = np.zeros(4, dtype='float64')
    result = np.zeros(4, dtype='float64')
    temp[0] = np.dot(quat[1:], pos)
    temp[1:] = np.cross(quat[1:], pos)+quat[0]*pos
    # result[0] = np.dot(temp[1:],-quat[1:])+temp[0]*quat[0]
    result[1:] = (
                  np.cross(temp[1:], -quat[1:]) +
                  temp[0]*quat[1:]+quat[0]*temp[1:])
    return result[1:]


def RotateQuat(quat1, quat2):
    """
    Rotates quat 2 by quat1

    Arguments:
        quat1 - array (q0, qx, qy, qz)
        quat2 - array (q0, qx, qy, qz)
    """
    quat1_conj = np.zeros(4, dtype='float64')
    quat1_conj[0] = quat1[0]
    quat1_conj[1:] = -quat1[1:]
    return QuatMult(QuatMult(quat1, quat2), quat1_conj)