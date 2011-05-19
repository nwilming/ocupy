#!/usr/bin/env python
"""A helper module that collects some useful auxillary functions."""

from scipy.misc import toimage, fromimage
from numpy import asarray
import numpy as np


def imresize(arr, newsize, interp='bicubic', mode=None):
    """
    resize a matrix to the desired dimensions. May not always lead to best
    behavior at borders. If possible, fixation density maps and features
    should directly be computed in the desired size.

    Parameters
    ----------
    arr : array_like
        Input data, in any form that can be converted to an array with at
        least 2 dimensions
    newsize : 2 element tupel
        newsize[0] is the new height, newsize[1] the new width of the image
    interp : string
        specifies the interpolation method. Possible values are nearest, 
        bilinear, and bicubic. Defaults to bicubic.
    mode : string
        specifies the mode with which arr is transformed to an image. Per
        default, if arr is a valid (N,3) byte-array giving the RGB values
        (from 0 to 255) then mode='P'. For 2D arrays, the data type of the
        values is used.

    Returns
    -------
    out : ndarray
        The resized version of the input array
    """
    newsize = list(newsize)
    newsize.reverse()
    newsize = tuple(newsize)
    arr = asarray(arr)
    func = {'nearest':0, 'bilinear':2, 'bicubic':3, 'cubic':3}
    if not mode and arr.ndim == 2:
        mode = arr.dtype.kind.upper()
    img = toimage(arr, mode=mode)

    img = img.resize(newsize, resample = func[interp])
    return fromimage(img)

    
def randsample(vec, nr_samples):
    """
    Draws nr_samples random samples from vec.
    """
    return np.random.permutation(vec)[0:nr_samples]

def ismember(ar1, ar2): 
    """ 
    A setmember1d, which works for arrays with duplicate values 
    """
    return np.in1d(ar1, ar2)

def calc_resize_factor(prediction, image_size):
    """
    Calculates how much prediction.shape and image_size differ.
    """
    resize_factor_x = prediction.shape[1] / float(image_size[1])
    resize_factor_y = prediction.shape[0] / float(image_size[0])
    if abs(resize_factor_x - resize_factor_y) > 1.0/image_size[1] :
        raise RuntimeError("""The aspect ratio of the fixations does not
                              match with the prediction: %f vs. %f"""
                              %(resize_factor_y, resize_factor_x))
    return (resize_factor_y, resize_factor_x)
    
def dict_2_mat(data, fill = True):
    """
    Creates a numpy array from a dictionary with only integers as keys and
    numpy arrays as values. Dimension 0 of the resulting array is formed from
    data.keys(). Missing values in keys can be filled up with np.nan (default)
    or ignored.

    Parameters
    ----------
    data : dict
        a dictionary with integers as keys and array-likes of the same shape
        as values
    fill : boolean
        flag specifying if the resulting matrix will keep a correspondence
        between dictionary keys and matrix indices by filling up missing keys
        with matrices of NaNs. Defaults to True

    Returns
    -------
    numpy array with one more dimension than the values of the input dict
    """
    if any([type(k) != int for k in data.keys()]):
        raise RuntimeError("Dictionary cannot be converted to matrix, " +
                            "not all keys are ints")
    base_shape = np.array(data.values()[0]).shape
    result_shape = list(base_shape)
    if fill:
        result_shape.insert(0, max(data.keys()) + 1)
    else:
        result_shape.insert(0, len(data.keys()))
    result = np.empty(result_shape) + np.nan
        
    for (i, (k, v)) in enumerate(data.items()):
        v = np.array(v)
        if v.shape != base_shape:
            raise RuntimeError("Dictionary cannot be converted to matrix, " +
                                        "not all values have same dimensions")
        result[fill and [k][0] or [i][0]] = v
    return result
        
def dict_fun(data, function):
    """
    Apply a function to all values in a dictionary, return a dictionary with
    results.

    Parameters
    ----------
    data : dict
        a dictionary whose values are adequate input to the second argument
        of this function. 
    function : function
        a function that takes one argument

    Returns
    -------
    a dictionary with the same keys as data, such that
    result[key] = function(data[key])
    """
    return dict((k, function(v)) for k, v in data.items())
