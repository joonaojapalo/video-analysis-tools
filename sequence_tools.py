__all__ = ["find_by_idx"]

def find_by_idx(frame, pose_idx):
    for obj in frame["objs"]:
        if obj["idx"] == pose_idx:
            return obj


def select_sequence_idx(sequence, pose_idx):
    return [find_by_idx(frame, pose_idx) for frame in sequence]
