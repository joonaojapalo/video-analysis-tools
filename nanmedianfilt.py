import numpy as np

def nanmedianfilt(x, window):
    n = x.shape[0]
    output = np.zeros(n)
    half_win, r = divmod(window, 2)

    if r != 1:
        raise ValueError("window length must be odd: %i" % window)

    for i in range(n):
        a = max(i - half_win, 0)
        b = min(i + half_win, n)
        output[i] = np.nanmedian(x[a:b])

    return output
