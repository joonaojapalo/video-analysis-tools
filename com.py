import numpy as np

from keypoints import KEYPOINTS

segment_groups = {
    "hands": [
        "left-arm",
        "right-arm",
        "left-forearm",
        "right-forearm"
    ],
    "legs": [
        "left-thigh",
        "right-thigh",
        "left-shank",
        "right-shank"
    ],
    "forearms": [
        "left-forearm",
        "right-forearm"
    ],
    "shanks": [
        "left-shank",
        "right-shank"
    ]
}


def com_interp(p_prox, p_dist, ratio):
    return ratio * p_prox + (1-ratio) * p_dist


def keypoint(world_pos, name):
    return world_pos[:, 3*KEYPOINTS[name]:3*KEYPOINTS[name] + 3]


def compute_segment_com(world_pos):
    p_lsh = keypoint(world_pos, "LShoulder")
    p_rsh = keypoint(world_pos, "RShoulder")
    p_lhip = keypoint(world_pos, "LHip")
    p_rhip = keypoint(world_pos, "RHip")
    p_trunk = com_interp((p_lsh + p_rsh) / 2, (p_lhip + p_rhip) / 2, 0.495)
    p_head = com_interp(keypoint(world_pos, "Head"),
                        keypoint(world_pos, "Neck"), 0.517)
    p_larm = com_interp(keypoint(world_pos, "LShoulder"),
                        keypoint(world_pos, "LElbow"), 0.436)
    p_rarm = com_interp(keypoint(world_pos, "RShoulder"),
                        keypoint(world_pos, "RElbow"), 0.436)
    p_lfore = com_interp(keypoint(world_pos, "LElbow"),
                         keypoint(world_pos, "LWrist"), 0.430)
    p_rfore = com_interp(keypoint(world_pos, "RElbow"),
                         keypoint(world_pos, "RWrist"), 0.430)
    p_lthigh = com_interp(keypoint(world_pos, "LHip"),
                          keypoint(world_pos, "LKnee"), 0.433)
    p_rthigh = com_interp(keypoint(world_pos, "RHip"),
                          keypoint(world_pos, "RKnee"), 0.433)
    p_rshank = com_interp(keypoint(world_pos, "RKnee"),
                          keypoint(world_pos, "RAnkle"), 0.433)
    p_lshank = com_interp(keypoint(world_pos, "LKnee"),
                          keypoint(world_pos, "LAnkle"), 0.433)
    return (p_trunk, p_head, p_larm, p_rarm, p_lfore, p_rfore, p_lthigh, p_rthigh, p_rshank, p_lshank)


def compute(world_pos, exclude=[]):
    """Compute CoM from 3D position coordinates using model by (Dempster, 1955) and
    adjusted by (Clauser et al., 1969).

    Parameters:
    world_pos (np.array)    : body segment joint positions (Halpe/26 keypoints)
    exclude (list(str))     : segments to exclude: "hands, legs, forearms, shanks"
    """
    # trunk_weight = 0.678
    # lower_extr_weight = 0.161
    # rprox_trunk = 0.626
    # rprox_lower_extr = 0.447

    (
        p_trunk,
        p_head,
        p_larm,
        p_rarm,
        p_lfore,
        p_rfore,
        p_lthigh,
        p_rthigh,
        p_rshank,
        p_lshank
    ) = compute_segment_com(world_pos)

    segments = {
        "head": (p_head, 0.0810),
        "trunk": (p_trunk, 0.4970),
        "left-arm": (p_larm, 0.0280),
        "right-arm": (p_rarm, 0.0280),
        "left-forearm": (p_lfore, 0.0160),
        "right-forearm": (p_rfore, 0.0160),
        "left-thigh": (p_lthigh, 0.1000),
        "right-thigh": (p_rthigh, 0.1000),
        "left-shank": (p_lshank, 0.0465),
        "right-shank": (p_rshank, 0.0465)
    }

    missing_segments = []
    for segname, (pos, weight) in segments.items():
        segnan = np.isnan(pos)
        if segnan.any():
            nanframes = np.where(segnan)[0]
            missing = (
                segname,
                min(nanframes),
                max(nanframes),
                segnan.any(1).sum(),
            )
            missing_segments.append(missing)

    if len(missing_segments):
        print("Segments with missing data:")
        for segname, f0, f1, total in missing_segments:
            print("  - %s (frames: %i on interval %i..%i)" %
                  (segname, total, f0, f1))
        print()

    comarr = np.zeros([world_pos.shape[0], 3])

    # apply exclusion list

    if exclude:
        print("Excluding segments:")
        for exclude_group_name in exclude:
            for exclude_name in segment_groups.get(exclude_group_name):
                del segments[exclude_name]
                print("  - %s" % exclude_name)
        print()

    weight_total = sum(s[1] for s in segments.values())

    # compute CoM as weighted mean
    for (pos, weight) in segments.values():
        comarr += pos * weight / weight_total

    # calculate segment CoMs
    #   = head, arms & trunk
    #   = greater trochanter (proximal) to glenohumeral joint (distal)
    # TODO: com_trunk_cam1 = segmentEndsCam1(:,1:2) + rprox_trunk * (segmentEndsCam1(:,3:4) - segmentEndsCam1(:,1:2));

    # LowerExtremity = greater trochanger (proximal) to medial malleolus (distal)
    # TODO: com_lower_extr_cam1 = segmentEndsCam1(:,1:2) + rprox_lower_extr * (segmentEndsCam1(:,5:6) - segmentEndsCam1(:,1:2));

    # Whole body
    ### TODO: com = trunk_weight * com_trunk_cam1 + 2 * (lower_extr_weight * com_lower_extr_cam1);
    return comarr
