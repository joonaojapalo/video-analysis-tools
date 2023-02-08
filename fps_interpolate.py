import numpy as np


def _interpolate(first, second, coefficient, state=0):
    assert first.shape == second.shape
    N = first.shape[0]

    output_arr = []
    while state <= 1:
        output = np.zeros(N)
        for dimension in range(0, N // 3):
            # x & y
            for i in range(2):
                col = 3 * dimension + i
                output[col] = np.interp(state,
                                        [0, 1],
                                        [first[col], second[col]])

            # visibility
            visibility_col = 3 * dimension + 2
            output[visibility_col] = np.min([first[visibility_col],
                                             second[visibility_col]])

        # row
        output_arr.append(output)
        state += coefficient

    return output_arr, state % 1


def interpolate(pose_arr, input_fps=240, target_fps=720):
    """Linear interpolate input pose_arr to match target_fps.

    Parameters:
    pose_arr (np.array): [frames, 78] sized array, where 78 = 26 x 3
                      keypoint coordinates (x, y, visibility_score)

    Returns:
    np.array: Interpolated pose array.
    """
    state = 0
    coefficient = input_fps / target_fps
    interpolated = []

    # for each consequtive samples
    for previous, current in zip(pose_arr[:-1], pose_arr[1:]):
        interpolated_rows, state = _interpolate(
            previous, current, coefficient, state)
        interpolated.extend(interpolated_rows)
        previous = current

    if state != 0:
        # extrapolate last frame
        previous = pose_arr[-2]
        current = pose_arr[-1]
        dx = state * (previous - current)
        interpolated.append(current + dx)

    return np.array(interpolated)


if __name__ == "__main__":
    print(interpolate(np.arange(10), np.ones(10)*20), 0.5)
