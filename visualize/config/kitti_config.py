import numpy as np

class_list = ["Car"]#,"Pedestrian", "Cyclist"]

CLASS_NAME_TO_ID = {
    'Car': 0,
    'VanSUV': 0
    #'Pedestrian': 1,
    #'Cyclist': 2,
    #'Bicycle': 2
}

# Front side (of vehicle) Point Cloud boundary for BEV
boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

# Back back (of vehicle) Point Cloud boundary for BEV
boundary_back = {
    "minX": -50,
    "maxX": 0,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

BEV_WIDTH = 608  # across y axis -25m ~ 25m
BEV_HEIGHT = 608  # across x axis 0m ~ 50m

DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]

# Following parameters are calculated as an average from KITTI dataset for simplicity
#####################################################################################

Tr_velo_to_cam = np.array([ [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0]])

# cal mean from train set
R0 = np.array([ [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]])

P2 = np.array([[1687.3369140625, 0.0,965.43414055823814, 0.0],
               [0.0, 1783.428466796875, 684.4193604186803, 0.0],
               [0.0, 0.0, 1.0, 0.0]])

UDM = np.array([[1687.3369140625, 0.0, 965.43414055823814],
                    [0.0, 1783.428466796875, 684.4193604186803],
                                    [0.0, 0.0, 1.0]])

DM = np.array([[1844.1774422790927, 0.0, 964.42990523863864],
                    [0.0, 1841.5212239377258, 679.5331911948183],
                                    [0.0, 0.0, 1.0]])

D = np.array([[-0.2611312587700434, 0.0, 0.0, 0.0, 0.0]])

# R0_inv = np.linalg.inv(R0)
# Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
# P2_inv = np.linalg.pinv(P2)

#####################################################################################

