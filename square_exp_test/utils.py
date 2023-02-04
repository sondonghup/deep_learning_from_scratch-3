import numpy as np

def as_array(x: float) -> np.ndarray: # float를 np.ndarray로 바꿔주기 위해서
    if np.isscalar(x):
        return np.array(x)
    return x