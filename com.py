import numpy as np

from keypoints import KEYPOINTS


def compute(world_pos, gender="m"):
    """Compute CoM from 3D position coordinates using model by Dempster"""
    # trunk_weight = 0.678
    # lower_extr_weight = 0.161
    # rprox_trunk = 0.626
    # rprox_lower_extr = 0.447

    segments = [
        # (proximal_kp, distal_kp, com_distance_f, com_distance_m, weight_f, weight_m)
        (KEYPOINTS["Head"], KEYPOINTS["Head"], 0.0, 0.0, 0.0668, 0.0694),
        (KEYPOINTS["Neck"], KEYPOINTS["LHip"],
         0.4151, 0.4486, 0.4257 / 2, 0.4346 / 2),
        (KEYPOINTS["Neck"], KEYPOINTS["RHip"],
         0.4151, 0.4486, 0.4257 / 2, 0.4346 / 2),
        (KEYPOINTS["LHip"], KEYPOINTS["LKnee"],
         0.3612, 0.4095, 0.1478 / 2, 0.1416 / 2),
        (KEYPOINTS["RHip"], KEYPOINTS["RKnee"],
         0.3612, 0.4095, 0.1478 / 2, 0.1416 / 2),
        (KEYPOINTS["LKnee"], KEYPOINTS["LAnkle"], 0.4416,
         0.4459, 0.0481 + 0.0129, 0.0433 + 0.0137),
        (KEYPOINTS["RKnee"], KEYPOINTS["RAnkle"], 0.4416,
         0.4459, 0.0481 + 0.0129, 0.0433 + 0.0137),
        (KEYPOINTS["LShoulder"], KEYPOINTS["LElbow"],
         0.5754, 0.5772, 0.0255, 0.0271),
        (KEYPOINTS["RShoulder"], KEYPOINTS["RElbow"],
         0.5754, 0.5772, 0.0255, 0.0271),
        (KEYPOINTS["LElbow"], KEYPOINTS["LWrist"], 0.4559,
         0.4574, 0.0138 + 0.0056, 0.0162 + 0.0061),
        (KEYPOINTS["RElbow"], KEYPOINTS["RWrist"], 0.4559,
         0.4574, 0.0138 + 0.0056, 0.0162 + 0.0061),
    ]

    # TODO: check availabile segments
    avail_segments = [] # [[HEAD, Nenck-LHip]]
    for (kp0, kp1, com_f, com_m, w_f, w_m) in segments:
        p0 = world_pos[:, 3*kp0:3*kp0 + 3]
        p1 = world_pos[:, 3*kp1:3*kp1 + 3]
        

    comarr = np.zeros([world_pos.shape[0], 3])

    # (x,y)
    weight_total_f = sum(s[4] for s in segments)
    weight_total_m = sum(s[5] for s in segments)
    w = 0.0
    for (kp0, kp1, com_f, com_m, w_f, w_m) in segments:
        p0 = world_pos[:, 3*kp0:3*kp0 + 3]
        p1 = world_pos[:, 3*kp1:3*kp1 + 3]
        comarr += (p0 + (p1 - p0) * com_f) * w_f / weight_total_f
        w += w_f

    # calculate segment CoMs
    #   = head, arms & trunk
    #   = greater trochanter (proximal) to glenohumeral joint (distal)
    # TODO: com_trunk_cam1 = segmentEndsCam1(:,1:2) + rprox_trunk * (segmentEndsCam1(:,3:4) - segmentEndsCam1(:,1:2));

    # LowerExtremity = greater trochanger (proximal) to medial malleolus (distal)
    # TODO: com_lower_extr_cam1 = segmentEndsCam1(:,1:2) + rprox_lower_extr * (segmentEndsCam1(:,5:6) - segmentEndsCam1(:,1:2));

    # Whole body
    ### TODO: com = trunk_weight * com_trunk_cam1 + 2 * (lower_extr_weight * com_lower_extr_cam1);
    return comarr
