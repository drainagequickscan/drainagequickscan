import numpy as np


def rounder(n, round_percentage=0.1):
    if np.isnan(n):
        return np.nan
    elif n == 0.:
        return 0.
    else:
        return np.round(n, -1 * int((np.log10(round_percentage * np.abs(n)))))

