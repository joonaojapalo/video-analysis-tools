from pathlib import Path
import numpy as np

class OutputWriter:
    def __init__(self, path) -> None:
        self.path = Path(path)

    def _ensure_path(self):
        # ensure directory
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, world_pos):
        if self.path.is_file():
            print(f"File exists, overwriting: {self.path}...")

        self._ensure_path()
        self.write_data(self.path, world_pos)

    def write_com(self, data):
        self._ensure_path()
        self.write_com_data(self.get_com_path(), data)

    def write_data(self, path, world_pos):
        raise NotImplementedError
    
    def write_com_data(self, com_pos):
        raise NotImplementedError

    def get_path(self):
        raise NotImplementedError

    def get_com_path(self):
        raise NotImplementedError


class NumpyOutputWriter (OutputWriter):
    def write_data(self, path, world_pos):
        np.save(str(path), world_pos)
    
    def write_com_data(self, path, com_pos):
        np.save(str(path), com_pos)

    def _get_com_path(self):
        com_file_name = f"{self.path.stem}-com{self.path.suffix}"
        return self.path.parent.joinpath(com_file_name)

    def get_com_path(self):
        com_path = self._get_com_path()
        return f"{com_path}.npy"

    def get_path(self):
        return f"{self.path}.npy"


csv_header = [
    "Nose_x",
    "Nose_y",
    "Nose_z",
    "LEye_x",
    "LEye_y",
    "LEye_z",
    "REye_x",
    "REye_y",
    "REye_z",
    "LEar_x",
    "LEar_y",
    "LEar_z",
    "REar_x",
    "REar_y",
    "REar_z",
    "LShoulder_x",
    "LShoulder_y",
    "LShoulder_z",
    "RShoulder_x",
    "RShoulder_y",
    "RShoulder_z",
    "LElbow_x",
    "LElbow_y",
    "LElbow_z",
    "RElbow_x",
    "RElbow_y",
    "RElbow_z",
    "LWrist_x",
    "LWrist_y",
    "LWrist_z",
    "RWrist_x",
    "RWrist_y",
    "RWrist_z",
    "LHip_x",
    "LHip_y",
    "LHip_z",
    "RHip_x",
    "RHip_y",
    "RHip_z",
    "LKnee_x",
    "LKnee_y",
    "LKnee_z",
    "RKnee_x",
    "RKnee_y",
    "RKnee_z",
    "LAnkle_x",
    "LAnkle_y",
    "LAnkle_z",
    "RAnkle_x",
    "RAnkle_y",
    "RAnkle_z",
    "Head_x",
    "Head_y",
    "Head_z",
    "Neck_x",
    "Neck_y",
    "Neck_z",
    "Hip_x",
    "Hip_y",
    "Hip_z",
    "LBigToe_x",
    "LBigToe_y",
    "LBigToe_z",
    "RBigToe_x",
    "RBigToe_y",
    "RBigToe_z",
    "LSmallToe_x",
    "LSmallToe_y",
    "LSmallToe_z",
    "RSmallToe_x",
    "RSmallToe_y",
    "RSmallToe_z",
    "LHeel_x",
    "LHeel_y",
    "LHeel_z",
    "RHeel_x",
    "RHeel_y",
    "RHeel_z"
]

csv_com_header = ["CoM_x", "CoM_y", "CoM_z"]
EOL = "\n"

def format_value(v):
    if np.isnan(v):
        return ''
    else:
        return str(v)

class CSVOuputWriter (OutputWriter):
    DELIM = ";"

    def write_data(self, basepath, world_pos):
        with open(self.get_path(), "w") as f:
            f.write(self.DELIM.join(csv_header) + EOL)

            for l in world_pos:
                f.write(self.DELIM.join(map(format_value, l)) + EOL)

    def write_com_data(self, path, com_pos):
        with open(self.get_com_path(), "w") as f:
            f.write(self.DELIM.join(csv_com_header) + EOL)

            for l in com_pos:
                f.write(self.DELIM.join(map(format_value, l)) + EOL)

    def _get_com_path(self):
        com_file_name = f"{self.path.stem}-com{self.path.suffix}"
        return self.path.parent.joinpath(com_file_name)

    def get_com_path(self):
        com_path = self._get_com_path()
        return f"{com_path}.csv"

    def get_path(self):
        return f"{self.path}.csv"
