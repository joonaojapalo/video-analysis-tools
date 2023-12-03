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


class InvalidCenterOfMassModelException (Exception):
    pass


class InvalidSegmentException (Exception):
    pass


class SegmentCenterOfMass:
    def __init__(self, name: str, position, relative_weight: float) -> None:
        self.name = name
        self.pos = position
        self.relative_weight = relative_weight


class DempsterCenterOfMassSegment (SegmentCenterOfMass):
    weights = {
        "head": 0.0810,
        "trunk": 0.4970,
        "left-arm": 0.0280,
        "right-arm": 0.0280,
        "left-forearm": 0.0160,
        "right-forearm": 0.0160,
        "left-thigh": 0.1000,
        "right-thigh": 0.1000,
        "left-shank": 0.0465,
        "right-shank": 0.0465,
    }

    def __init__(self, name: str, position) -> None:
        if name not in self.weights:
            raise InvalidSegmentException(name)

        super().__init__(name, position, self.weights[name])


class CenterOfMassModel:
    def compute_segment_com(self, world_pos):
        raise NotImplementedError()


class DempsterAlphapose (CenterOfMassModel):
    def compute_segment_com(self, world_pos):
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

        # Dempster (1955) body segment positions and weights
        return [
            DempsterCenterOfMassSegment("head", p_head),
            DempsterCenterOfMassSegment("trunk", p_trunk),
            DempsterCenterOfMassSegment("left-arm", p_larm),
            DempsterCenterOfMassSegment("right-arm", p_rarm),
            DempsterCenterOfMassSegment("left-forearm", p_lfore),
            DempsterCenterOfMassSegment("right-forearm", p_rfore),
            DempsterCenterOfMassSegment("left-thigh", p_lthigh),
            DempsterCenterOfMassSegment("right-thigh", p_rthigh),
            DempsterCenterOfMassSegment("left-shank", p_lshank),
            DempsterCenterOfMassSegment("right-shank", p_rshank),
        ]


class DempsterKIHU (CenterOfMassModel):
    def compute_segment_com(self, world_pos):
        p_lsh = keypoint(world_pos, "LShoulder")
        p_head = keypoint(world_pos, "Head")
        p_rsh = keypoint(world_pos, "RShoulder")
        p_lhip = keypoint(world_pos, "LHip")
        p_rhip = keypoint(world_pos, "RHip")
        p_trunk = com_interp((p_lsh + p_rsh) / 2, (p_lhip + p_rhip) / 2, 0.495)
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

        # Dempster (1955) body segment positions and weights
        return [
            DempsterCenterOfMassSegment("head", p_head),
            DempsterCenterOfMassSegment("trunk", p_trunk),
            DempsterCenterOfMassSegment("left-arm", p_larm),
            DempsterCenterOfMassSegment("right-arm", p_rarm),
            DempsterCenterOfMassSegment("left-forearm", p_lfore),
            DempsterCenterOfMassSegment("right-forearm", p_rfore),
            DempsterCenterOfMassSegment("left-thigh", p_lthigh),
            DempsterCenterOfMassSegment("right-thigh", p_rthigh),
            DempsterCenterOfMassSegment("left-shank", p_lshank),
            DempsterCenterOfMassSegment("right-shank", p_rshank),
        ]


def build_model(model_name: str):
    if model_name == "dempster-alphapose":
        return DempsterAlphapose()
    elif model_name == "dempster-kihu":
        return DempsterKIHU()
    else:
        raise InvalidCenterOfMassModelException()


def compute(world_pos, exclude=[], model_name="dempster-alphapose"):
    """Compute CoM from 3D position coordinates using model by (Dempster, 1955) and
    adjusted by (Clauser et al., 1969).

    Parameters:
    world_pos (np.array)    : body segment joint positions (Halpe/26 keypoints)
    exclude (list(str))     : segments to exclude: "hands, legs, forearms, shanks"
    model_name (str)        : center-of-mass model: "dempster-alphapose" | "dempster-kihu"
    """
    # trunk_weight = 0.678
    # lower_extr_weight = 0.161
    # rprox_trunk = 0.626
    # rprox_lower_extr = 0.447

    # build CoM model
    model = build_model(model_name)
    com_segments = model.compute_segment_com(world_pos)

    missing_segments = []
    for com_segment in com_segments:
        segnan = np.isnan(com_segment.pos)
        if segnan.any():
            nanframes = np.where(segnan)[0]
            missing = (
                com_segment.name,
                min(nanframes),
                max(nanframes),
                segnan.any(1).sum(),
            )
            missing_segments.append(missing)

    if len(missing_segments):
        print("Segments with missing data:")
        for name, f0, f1, total in missing_segments:
            print("  - %s (frames: %i on interval %i..%i)" %
                  (name, total, f0, f1))
        print()

    comarr = np.zeros([world_pos.shape[0], 3])

    # apply exclusion list
    if exclude:
        print("Excluding segments:")
        for exclude_group_name in exclude:
            for exclude_name in segment_groups.get(exclude_group_name):
                del com_segments[exclude_name]
                print("  - %s" % exclude_name)
        print()

    weight_total = sum(s.relative_weight for s in com_segments)

    # compute CoM as weighted mean
    for segment in com_segments:
        comarr += segment.pos * segment.relative_weight / weight_total

    # calculate segment CoMs
    #   = head, arms & trunk
    #   = greater trochanter (proximal) to glenohumeral joint (distal)
    # TODO: com_trunk_cam1 = segmentEndsCam1(:,1:2) + rprox_trunk * (segmentEndsCam1(:,3:4) - segmentEndsCam1(:,1:2));

    # LowerExtremity = greater trochanger (proximal) to medial malleolus (distal)
    # TODO: com_lower_extr_cam1 = segmentEndsCam1(:,1:2) + rprox_lower_extr * (segmentEndsCam1(:,5:6) - segmentEndsCam1(:,1:2));

    # Whole body
    ### TODO: com = trunk_weight * com_trunk_cam1 + 2 * (lower_extr_weight * com_lower_extr_cam1);
    return comarr
