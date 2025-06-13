import numpy as np
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils import check_array, check_random_state

def make_t(n_samples:int, shuffle=True, noise=None, random_state=None):
    """Make lines intersecting in a T formation

    :param n_samples: _description_
    :type n_samples: _type_
    :param shuffle: _description_, defaults to True
    :type shuffle: bool, optional

    """

    generator = check_random_state(random_state)

    n_samples_stem  = n_samples // 2
    n_samples_cross = n_samples - n_samples_stem

    stem_x = np.linspace(0.5, 0.5, n_samples_stem)
    stem_y = np.linspace(0.0, 1.0, n_samples_stem)
    cross_x = np.linspace(0.0, 1.0, n_samples_cross)
    cross_y = np.linspace(1.0, 1.0, n_samples_cross)
    
    X = np.vstack(
        [np.append(stem_x, cross_x), np.append(stem_y, cross_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_stem, dtype=np.intp), np.ones(n_samples_cross, dtype=np.intp)]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y



def make_v(n_samples:int, shuffle=True, noise=None, random_state=None):
    """Make lines intersecting in a V formation

    :param n_samples: _description_
    :type n_samples: _type_
    :param shuffle: _description_, defaults to True
    :type shuffle: bool, optional

    """

    generator = check_random_state(random_state)

    n_samples_left  = n_samples // 2
    n_samples_right = n_samples - n_samples_left

    left_x = np.linspace(0.5, 0.0, n_samples_left)
    left_y = np.linspace(0.0, 1.0, n_samples_left)
    right_x = np.linspace(0.5, 1.0, n_samples_right)
    right_y = np.linspace(0.0, 1.0, n_samples_right)
    
    X = np.vstack(
        [np.append(left_x, right_x), np.append(left_y, right_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_left, dtype=np.intp), np.ones(n_samples_right, dtype=np.intp)]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y



def make_a(n_samples:int, shuffle=True, noise=None, random_state=None):
    """Make lines intersecting in a V formation

    :param n_samples: _description_
    :type n_samples: _type_
    :param shuffle: _description_, defaults to True
    :type shuffle: bool, optional

    """

    generator = check_random_state(random_state)

    n_samples_left  = 3 * n_samples // 8
    n_samples_right = 3 * n_samples // 8
    n_samples_cross = 2 * n_samples // 8

    left_x = np.linspace(0.5, 0.0, n_samples_left)
    left_y = np.linspace(1.0, 0.0, n_samples_left)
    right_x = np.linspace(0.5, 1.0, n_samples_right)
    right_y = np.linspace(1.0, 0.0, n_samples_right)
    cross_x = np.linspace(0.33, 0.66, n_samples_cross)
    cross_y = np.linspace(0.33, 0.33, n_samples_cross)
    
    X = np.vstack(
        [np.append(np.append(left_x, right_x), cross_x), np.append(np.append(left_y, right_y), cross_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_left, dtype=np.intp), np.ones(n_samples_right, dtype=np.intp), np.ones(n_samples_cross, dtype=np.intp)]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y


def make_duck_foot(n_samples:int, shuffle=True, noise=None, random_state=None):
    """Make lines intersecting in a V formation

    :param n_samples: _description_
    :type n_samples: _type_
    :param shuffle: _description_, defaults to True
    :type shuffle: bool, optional

    """

    generator = check_random_state(random_state)

    n_samples_left  = n_samples // 3
    n_samples_right = n_samples // 3
    n_samples_middle = n_samples // 3

    left_x = np.linspace(-0.25, 0.5, n_samples_left)
    left_y = np.linspace(0.75, 0.0, n_samples_left)
    right_x = np.linspace(0.5, 1.25, n_samples_right)
    right_y = np.linspace(0.0, 0.75, n_samples_right)
    middle_x = np.linspace(0.5, 0.5, n_samples_middle)
    middle_y = np.linspace(0.0, 1.0, n_samples_middle)
    
    X = np.vstack(
        [np.append(np.append(left_x, right_x), middle_x), np.append(np.append(left_y, right_y), middle_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_left, dtype=np.intp), np.ones(n_samples_right, dtype=np.intp), np.ones(n_samples_middle, dtype=np.intp)]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y