import csv
import pathlib
import keypoints
import numpy as np

def read_offset_table(offset_table_path: str):
    """ Offset table stores data correction information of how many frames
        each landmark column in input data should be offset.
    """
    table = {}
    p = pathlib.Path(offset_table_path)

    if not p.exists():
        raise Exception("Offset table doesn't exist: %s" % (offset_table_path,))

    # Open the CSV file for reading
    with open(p, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            if len(row) != 2:
                continue

            landmark_name, offset = row

            if landmark_name not in keypoints.KEYPOINTS:
                raise Exception("Invalid landmark name '%s' in offset table: %s" % (landmark_name, offset_table_path))

            table[landmark_name] = int(offset)

    return table



def arr_shift(arr, col_idx, offset):
    """Shift column col_idx of array by number of rows (offset)
    """
    if offset == 0:
        return arr

    shf = np.copy(arr)
    if offset > 0:
        shf[offset:, col_idx] = arr[:-offset, col_idx]
        shf[:offset, col_idx] = arr[0, col_idx]
    else:
        offset = abs(offset)
        shf[:-offset, col_idx] = arr[offset:, col_idx]
        shf[-offset:, col_idx] = arr[-1, col_idx]

    return shf


def apply_offset_table(keypoint_arr, offset_table_path):
    """apply landmark offsets to keypoint_arr
    """
    offset_table = read_offset_table(offset_table_path)

    for lm, offset in offset_table.items():
        for dim in range(3):
            # find landmark column index (x, y & visibility) 
            col_idx = 3 * keypoints.KEYPOINTS[lm] + dim
            keypoint_arr = arr_shift(keypoint_arr, col_idx, offset)

    return keypoint_arr