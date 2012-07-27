#!/usr/bin/env python
"""A helper module that collects some useful auxillary functions."""

from numpy import asarray
import numpy as np
import cPickle

have_image_library=True
try:
    from scipy.misc import toimage, fromimage #@UnresolvedImport
except ImportError:
    have_image_library=False

def isiterable(some_object):
    try:
        iter(some_object)
    except TypeError:
        return False
    return True


if have_image_library:
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
    Creates a NumPy array from a dictionary with only integers as keys and
    NumPy arrays as values. Dimension 0 of the resulting array is formed from
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

def snip_string_middle(string, max_len=20, snip_string='...'):
    """
    >>> snip_string_middle('this is long', 8)
    'th...ong'
    >>> snip_string_middle('this is long', 12)
    'this is long'
    >>> snip_string_middle('this is long', 8, '~')
    'thi~long'
    

    """
    #warn('use snip_string() instead', DeprecationWarning)
    if len(string) <= max_len:
        new_string = string
    else:
        visible_len = (max_len - len(snip_string))
        start_len = visible_len//2
        end_len = visible_len-start_len
        
        new_string = string[0:start_len]+ snip_string + string[-end_len:]
    
    return new_string
   
def snip_string(string, max_len=20, snip_string='...', snip_point=0.5):
    """
    Snips a string so that it is no longer than max_len, replacing deleted
    characters with the snip_string.
    The snip is done at snip_point, which is a fraction between 0 and 1,
    indicating relatively where along the string to snip. snip_point of
    0.5 would be the middle.
    >>> snip_string('this is long', 8)
    'this ...'
    >>> snip_string('this is long', 8, snip_point=0.5)
    'th...ong'
    >>> snip_string('this is long', 12)
    'this is long'
    >>> snip_string('this is long', 8, '~')
    'this is~'
    >>> snip_string('this is long', 8, '~', 0.5)
    'thi~long'
    
    """
    if len(string) <= max_len:
        new_string = string
    else:
        visible_len = (max_len - len(snip_string))
        start_len = int(visible_len*snip_point)
        end_len = visible_len-start_len
        
        new_string = string[0:start_len]+ snip_string
        if end_len > 0:
            new_string += string[-end_len:]
    
    return new_string 

def find_common_beginning(string_list, boundary_char = None):
    """Given a list of strings, finds finds the longest string that is common
    to the *beginning* of all strings in the list.
    
    boundary_char defines a boundary that must be preserved, so that the
    common string removed must end with this char.
    """
    
    common=''
    
    # by definition there is nothing common to 1 item...
    if len(string_list) > 1:
        shortestLen = min([len(el) for el in string_list])
        
        for idx in range(shortestLen):
            chars = [s[idx] for s in string_list]
            if chars.count(chars[0]) != len(chars): # test if any chars differ
                break
            common+=chars[0]
    
        
    if boundary_char is not None:
        try:
            end_idx = common.rindex(boundary_char)
            common = common[0:end_idx+1]
        except ValueError:
            common = ''
    
    return common

def factorise_strings (string_list, boundary_char=None):
    """Given a list of strings, finds the longest string that is common
    to the *beginning* of all strings in the list and
    returns a new list whose elements lack this common beginning.
    
    boundary_char defines a boundary that must be preserved, so that the
    common string removed must end with this char.
    
    >>> cmn='something/to/begin with?'
    >>> blah=[cmn+'yes',cmn+'no',cmn+'?maybe']
    >>> (blee, bleecmn) = factorise_strings(blah)
    >>> blee
    ['yes', 'no', '?maybe']
    >>> bleecmn == cmn
    True
    
    >>> blah = ['de.uos.nbp.senhance', 'de.uos.nbp.heartFelt']
    >>> (blee, bleecmn) = factorise_strings(blah, '.')
    >>> blee
    ['senhance', 'heartFelt']
    >>> bleecmn
    'de.uos.nbp.'
    
    >>> blah = ['/some/deep/dir/subdir', '/some/deep/other/dir', '/some/deep/other/dir2']
    >>> (blee, bleecmn) = factorise_strings(blah, '/')
    >>> blee
    ['dir/subdir', 'other/dir', 'other/dir2']
    >>> bleecmn
    '/some/deep/'
    
    >>> blah = ['/net/store/nbp/heartFelt/data/ecg/emotive_interoception/p20/2012-01-27T09.01.14-ecg.csv', '/net/store/nbp/heartFelt/data/ecg/emotive_interoception/p21/2012-01-27T11.03.08-ecg.csv', '/net/store/nbp/heartFelt/data/ecg/emotive_interoception/p23/2012-01-31T12.02.55-ecg.csv']
    >>> (blee, bleecmn) = factorise_strings(blah, '/')
    >>> bleecmn
    '/net/store/nbp/heartFelt/data/ecg/emotive_interoception/'
    
    rmuil 2012/02/01
    """
    
    cmn = find_common_beginning(string_list, boundary_char)
    
    new_list = [el[len(cmn):] for el in string_list]

    return (new_list, cmn)

class Memoize:
    """
    Memoize with mutable arguments
    """
    def __init__(self, function):
        self.function = function
        self.memory = {}

    def __call__(self, *args, **kwargs):
        hash_str = cPickle.dumps(args) + cPickle.dumps(kwargs)
        if not hash_str in self.memory:
            self.memory[hash_str] = self.function(*args, **kwargs)
        return self.memory[hash_str]


if __name__ == '__main__':
    import doctest
    doctest.testmod()

