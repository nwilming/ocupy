#!/usr/bin/env python
"""This module implements different model evaluation measures."""

import inspect

import numpy as np

from ocupy import fixmat
from ocupy.fixmat import compute_fdm, NoFixations
from ocupy.utils import calc_resize_factor


# List of measures, initialized below
scores = []

def set_scores(score_list):
    """
    Changes list of measures that are used by prediction_scores.
    """
    global scores
    scores = score_list
        
def prediction_scores(prediction, fm, **kw):
    """
    Evaluates a prediction against fixations in a fixmat with different measures.
    
    The default measures which are used are AUC, NSS and KL-divergence. This 
    can be changed by setting the list of measures with set_scores.  
    As different measures need potentially different parameters, the kw
    dictionary can be used to pass arguments to measures. Every named 
    argument (except fm and prediction) of a measure that is included in 
    kw.keys() will be filled with the value stored in kw.
    Example:
    
    >>> prediction_scores(P, FM, ctr_loc = (y,x))
    
    In this case the AUC will be computed with control points (y,x), because
    the measure 'roc_model' has 'ctr_loc' as named argument.
    
    Input:
        prediction  :   2D numpy array
            The prediction that should be evaluated
        fm  :   Fixmat
            The eyetracking data to evaluate against
    Output:
        Tuple of prediction scores. The order of the scores is determined
        by order of measures.scores.
    """
    if prediction == None:
        return [np.NaN for measure in scores]
    results = []
    for measure in scores:
        (args, _, _, _) = inspect.getargspec(measure)
        if len(args)>2:
            # Filter dictionary, such that only the keys that are 
            # expected by the measure are in it
            mdict = {}
            [mdict.update({key:value}) for (key, value) in kw.iteritems() 
                if key in args]
            score = measure(prediction, fm, **mdict)
        else:
            score = measure(prediction, fm)
        results.append(score)
    return results

def funky_test_measure(prediction, fm, arg1 = 'bar', arg2='foo'):
    """
    Measure that can be used for testing
    """
    return arg1 + arg2
    
def evaluate_predictions(stimuli):
    results = {}
    for cat in stimuli:
        score = [prediction_scores(img['prediction'], img.fixations) 
            for img in cat]
        results.update({cat.category: score})
    return results

def kldiv_model(prediction, fm):
    """
    wraps kldiv functionality for model evaluation

    input:
        prediction: 2D matrix 
            the model salience map
        fm : fixmat 
            Should be filtered for the image corresponding to the prediction
    """
    (_, r_x) = calc_resize_factor(prediction, fm.image_size)
    q = np.array(prediction, copy=True)
    q -= np.min(q.flatten())
    q /= np.sum(q.flatten())
    return kldiv(None, q, distp = fm, scale_factor = r_x)

def kldiv(p, q, distp = None, distq = None, scale_factor = 1):
    """
    Computes the Kullback-Leibler divergence between two distributions.

    Parameters
        p : Matrix
            The first probability distribution
        q : Matrix
            The second probability distribution
        distp : fixmat
            If p is None, distp is used to compute a FDM which 
            is then taken as 1st probability distribution.
        distq : fixmat
            If q is None, distq is used to compute a FDM which is 
            then taken as 2dn probability distribution.
        scale_factor : double
            Determines the size of FDM computed from distq or distp.

    """
    assert q != None or distq != None, "Either q or distq have to be given"
    assert p != None or distp != None, "Either p or distp have to be given"

    try:
        if p == None: 
            p = compute_fdm(distp, scale_factor = scale_factor)
        if q == None:
            q = compute_fdm(distq, scale_factor = scale_factor)
    except NoFixations:
        return np.NaN

    q += np.finfo(q.dtype).eps
    p += np.finfo(p.dtype).eps 
    kl = np.sum( p * (np.log2(p / q)))
    return kl
  
def kldiv_cs_model(prediction, fm):
    """
    Computes Chao-Shen corrected KL-divergence between prediction 
    and fdm made from fixations in fm.
    
    Parameters :
        prediction : np.ndarray
            a fixation density map
        fm : FixMat object
    """
    # compute histogram of fixations needed for ChaoShen corrected kl-div
    # image category must exist (>-1) and image_size must be non-empty
    assert(len(fm.image_size) == 2 and (fm.image_size[0] > 0) and
        (fm.image_size[1] > 0))
    assert(-1 not in fm.category)
    # check whether fixmat contains fixations
    if len(fm.x) ==  0:
        return np.NaN
    (scale_factor, _) = calc_resize_factor(prediction, fm.image_size)
    # this specifies left edges of the histogram bins, i.e. fixations between
    # ]0 binedge[0]] are included. --> fixations are ceiled
    e_y = np.arange(0, np.round(scale_factor*fm.image_size[0]+1))
    e_x = np.arange(0, np.round(scale_factor*fm.image_size[1]+1))
    samples = np.array(zip((scale_factor*fm.y), (scale_factor*fm.x)))
    (fdm, _) = np.histogramdd(samples, (e_y, e_x))

    # compute ChaoShen corrected kl-div
    q = np.array(prediction, copy = True)
    q[q == 0] = np.finfo(q.dtype).eps
    q /= np.sum(q)
    (H, pa, la) = chao_shen(fdm)
    q = q[fdm > 0]
    cross_entropy = -np.sum((pa * np.log2(q)) / la)
    return (cross_entropy - H)
    

def chao_shen(q):
    """
    Computes some terms needed for the Chao-Shen KL correction.
    """
    yx = q[q > 0] # remove bins with zero counts
    n = np.sum(yx)
    p = yx.astype(float)/n
    f1 = np.sum(yx == 1) # number of singletons in the sample
    if f1 == n: # avoid C == 0
        f1 -= 1
    C = 1 - (f1/n) # estimated coverage of the sample
    pa = C * p  # coverage adjusted empirical frequencies
    la = (1 - (1 - pa) ** n)  # probability to see a bin (species) in the sample
    H = -np.sum((pa * np.log2(pa)) / la)
    return (H, pa, la)
 

def correlation_model(prediction, fm):
    """
    wraps numpy.corrcoef functionality for model evaluation

    input:
        prediction: 2D Matrix 
            the model salience map
        fm: fixmat 
            Used to compute a FDM to which the prediction is compared.
    """
    (_, r_x) = calc_resize_factor(prediction, fm.image_size)
    fdm = compute_fdm(fm, scale_factor = r_x)
    return np.corrcoef(fdm.flatten(), prediction.flatten())[0,1]
    
    
def nss_model(prediction, fm):
    """
    wraps nss functionality for model evaluation
    
    input:
        prediction: 2D matrix
            the model salience map
        fm : fixmat
            Fixations that define the actuals
    """
    (r_y, r_x) = calc_resize_factor(prediction, fm.image_size)
    fix = ((np.array(fm.y-1)*r_y).astype(int),
                            (np.array(fm.x-1)*r_x).astype(int))
    return nss(prediction, fix)


def nss(prediction, fix):
    """
    Compute the normalized scanpath salience

    input:
        fix : list, l[0] contains y, l[1] contains x
    """

    prediction = prediction - np.mean(prediction)
    prediction = prediction / np.std(prediction)
    return np.mean(prediction[fix[0], fix[1]])


def roc_model(prediction, fm, ctr_loc = None, ctr_size = None):
    """
    wraps roc functionality for model evaluation
    
    Parameters:
        prediction: 2D array
            the model salience map
        fm : fixmat
            Fixations that define locations of the actuals
        ctr_loc : tuple of (y.x) coordinates, optional
            Allows to specify control points for spatial 
            bias correction
        ctr_size : two element tuple, optional
            Specifies the assumed image size of the control locations,
            defaults to fm.image_size
     """

    # check if prediction is a valid numpy array
    assert type(prediction) == np.ndarray
    # check whether scaling preserved aspect ratio
    (r_y, r_x) = calc_resize_factor(prediction, fm.image_size)
    # read out values in the fdm at actual fixation locations
    # .astype(int) floors numbers in np.array
    y_index = (r_y * np.array(fm.y-1)).astype(int)
    x_index = (r_x * np.array(fm.x-1)).astype(int)
    actuals = prediction[y_index, x_index]
    if not ctr_loc: 
        xc = np.random.randint(0, prediction.shape[1], 1000)
        yc = np.random.randint(0, prediction.shape[0], 1000)
        ctr_loc = (yc.astype(int), xc.astype(int))
    else:
        if not ctr_size:
            ctr_size = fm.image_size
        else:
            (r_y, r_x) = calc_resize_factor(prediction, ctr_size)
        ctr_loc = ((r_y * np.array(ctr_loc[0])).astype(int), 
                   (r_x * np.array(ctr_loc[1])).astype(int))
    controls = prediction[ctr_loc[0], ctr_loc[1]]

    return fast_roc(actuals, controls)[0]
    

def fast_roc(actuals, controls):
    """
    approximates the area under the roc curve for sets of actuals and controls.
    Uses all values appearing in actuals as thresholds and lower sum 
    interpolation. Also returns arrays of the true positive rate and the false
    positive rate that can be used for plotting the roc curve.
    
    Parameters:
        actuals : list
            A list of numeric values for positive observations.
        controls : list
            A list of numeric values for negative observations.
    """

    actuals = np.ravel(actuals)
    controls = np.ravel(controls)
    if np.isnan(actuals).any():
        raise RuntimeError('NaN found in actuals')
    if np.isnan(controls).any():
        raise RuntimeError('NaN found in controls')

    thresholds = np.hstack([-np.inf, np.unique(actuals), np.inf])[::-1]
    true_pos_rate = np.empty(thresholds.size)
    false_pos_rate = np.empty(thresholds.size)
    num_act = float(len(actuals))
    num_ctr = float(len(controls))

    for i, value in enumerate(thresholds):
        true_pos_rate[i] = (actuals >= value).sum() / num_act
        false_pos_rate[i] = (controls >= value).sum() / num_ctr
    auc = np.dot(np.diff(false_pos_rate), true_pos_rate[0:-1])
    # treat cases where TPR of one is not reached before FPR of one
    # by using trapezoidal integration for the last segment
    # (add the missing triangle)
    if false_pos_rate[-2] == 1:
        auc += ((1-true_pos_rate[-3])*.5*(1-false_pos_rate[-3]))
    return (auc, true_pos_rate, false_pos_rate)


def exact_roc(actuals, controls):
    """
    computes the area under the roc curve for separating to sets. Uses all
    possibl thresholds and trapezoidal interpolation. Also returns arrays of
    the true positive rate and the false positive rate.
    """

    actuals = np.ravel(actuals)
    controls = np.ravel(controls)
    if np.isnan(actuals).any():
        raise RuntimeError('NaN found in actuals')
    if np.isnan(controls).any():
        raise RuntimeError('NaN found in controls')

    thresholds = np.hstack([-np.inf,
        np.unique(np.concatenate((actuals,controls))), np.inf])[::-1]
    true_pos_rate = np.empty(thresholds.size)
    false_pos_rate = np.empty(thresholds.size)
    num_act = float(len(actuals))
    num_ctr = float(len(controls))

    for i, value in enumerate(thresholds):
        true_pos_rate[i] = (actuals >= value).sum() / num_act
        false_pos_rate[i] = (controls >= value).sum() / num_ctr
    auc = np.dot(np.diff(false_pos_rate),
            (true_pos_rate[0:-1]+true_pos_rate[1:])/2)
    return(auc, true_pos_rate, false_pos_rate)


def emd_model(prediction, fm):
    """
    wraps emd functionality for model evaluation
    
    requires:
        OpenCV python bindings
        
    input:
        prediction: the model salience map
        fm : fixmat filtered for the image corresponding to the prediction
    """
    (_, r_x) = calc_resize_factor(prediction, fm.image_size)
    gt = fixmat.compute_fdm(fm, scale_factor = r_x)
    return emd(prediction, gt)


def emd(prediction, ground_truth):
    """ 
    Compute the Eart Movers Distance between prediction and model.
    
    This implementation uses opencv for doing the actual work.
    Unfortunately, at the time of implementation only the SWIG
    bindings werer available and the numpy arrays have to
    converted by hand. This changes with opencv 2.1.  """
    import opencv

    if not (prediction.shape == ground_truth.shape):
        raise RuntimeError('Shapes of prediction and ground truth have' +
                           ' to be equal. They are: %s, %s'
                            %(str(prediction.shape), str(ground_truth.shape)))
    (x, y) = np.meshgrid(range(0, prediction.shape[1]),
                        range(0, prediction.shape[0]))    
    s1 = np.array([x.flatten(), y.flatten(), prediction.flatten()]).T
    s2 = np.array([x.flatten(), y.flatten(), ground_truth.flatten()]).T
    s1m = opencv.cvCreateMat(s1.shape[0], s2.shape[1], opencv.CV_32FC1)
    s2m = opencv.cvCreateMat(s1.shape[0], s2.shape[1], opencv.CV_32FC1)
    for r in range(0, s1.shape[0]):
        for c in range(0, s1.shape[1]):
            s1m[r, c] = float(s1[r, c])
            s2m[r, c] = float(s2[r, c])
    d = opencv.cvCalcEMD2(s1m, s2m, opencv.CV_DIST_L2)
    return d

# Default list of measures 
scores = [roc_model, nss_model, kldiv_model]


