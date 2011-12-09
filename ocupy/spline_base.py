import numpy as np
from scikits.learn import linear_model

from utils import Memoize

def fit3d(samples, e_x, e_y, e_z, remove_zeros = False, **kw):
    """Fits a 3D distribution with splines.

    Input:
        samples: Array
            Array of samples from a probability distribution
        e_x: Array
            Edges that define the events in the probability 
            distribution along the x direction. For example, 
            e_x[0] < samples[0] <= e_x[1] picks out all 
            samples that are associated with the first event.
        e_y: Array
            See e_x, but for the y direction.
        remove_zeros: Bool
            If True, events that are not observed will not 
            be part of the fitting process. If False, those 
            events will be modelled as finfo('float').eps 
        **kw: Arguments that are passed on to spline_bse1d.

    Returns:
        distribution: Array
            An array that gives an estimate of probability for 
            events defined by e.
        knots: Tuple of arrays
            Sequence of knots that were used for the spline basis (x,y) 
    """
    height, width, depth = len(e_y)-1, len(e_x)-1, len(e_z)-1 
    
    (p_est, _) = np.histogramdd(samples, (e_x, e_y, e_z))
    p_est = p_est/sum(p_est.flat)
    p_est = p_est.flatten()
    if remove_zeros:
        non_zero = ~(p_est == 0)
    else:
        non_zero = (p_est >= 0)
    basis = spline_base3d(width,height, depth, **kw)
    model = linear_model.BayesianRidge()
    model.fit(basis[:, non_zero].T, p_est[:,np.newaxis][non_zero,:])
    return (model.predict(basis.T).reshape((width, height, depth)), 
                p_est.reshape((width, height, depth)))
       
       
def fit2d(samples,e_x, e_y, remove_zeros = False, **kw):
    """Fits a 2D distribution with splines.

    Input:
        samples: Matrix or list of arrays 
            If matrix, it must be of size Nx2, where N is the number of
            observations. If list, it must contain two arrays of length
            N.
        e_x: Array
            Edges that define the events in the probability 
            distribution along the x direction. For example, 
            e_x[0] < samples[0] <= e_x[1] picks out all 
            samples that are associated with the first event.
        e_y: Array
            See e_x, but for the y direction.
        remove_zeros: Bool
            If True, events that are not observed will not 
            be part of the fitting process. If False, those 
            events will be modelled as finfo('float').eps 
        **kw: Arguments that are passed on to spline_bse1d.

    Returns:
        distribution: Array
            An array that gives an estimate of probability for 
            events defined by e.
        knots: Tuple of arrays
            Sequence of knots that were used for the spline basis (x,y) 
    """
    height = len(e_y)-1
    width = len(e_x)-1   
    (p_est, _) = np.histogramdd(samples, (e_x, e_y))
    # p_est contains x in dim 1 and y in dim 0
    shape = p_est.shape
    p_est = (p_est/sum(p_est.flat)).reshape(shape)
    mx =  p_est.sum(1)
    my = p_est.sum(0)
    # Transpose hist to have x in dim 0
    p_est = p_est.T.flatten()
    basis, knots = spline_base2d(width, height, marginal_x = mx, marginal_y = my, **kw)
    model = linear_model.BayesianRidge()
    if remove_zeros:
        non_zero = ~(p_est == 0)
        model.fit(basis[:, non_zero].T, p_est[:,np.newaxis][non_zero,:])
    else:
        non_zero = (p_est >= 0)
        p_est[:,np.newaxis][~non_zero,:] = np.finfo(float).eps
        model.fit(basis.T, p_est[:,np.newaxis])
    return (model.predict(basis.T).reshape((height, width)), 
            p_est.reshape((height, width)), knots)

def fit1d(samples, e, remove_zeros = False, **kw):
    """Fits a 1D distribution with splines.

    Input:
        samples: Array
            Array of samples from a probability distribution
        e: Array
            Edges that define the events in the probability 
            distribution. For example, e[0] < x <= e[1] is
            the range of values that are associated with the
            first event.
        **kw: Arguments that are passed on to spline_bse1d.

    Returns:
        distribution: Array
            An array that gives an estimate of probability for 
            events defined by e.
        knots: Array
            Sequence of knots that were used for the spline basis
    """
    samples = samples[~np.isnan(samples)]
    length = len(e)-1
    hist,_ = np.histogramdd(samples, (e,))
    hist = hist/sum(hist)
    basis, knots = spline_base1d(length, marginal = hist, **kw)
    non_zero = hist>0
    model = linear_model.BayesianRidge()
    if remove_zeros:
        model.fit(basis[non_zero, :], hist[:,np.newaxis][non_zero,:])
    else:
        hist[~non_zero] = np.finfo(float).eps
        model.fit(basis, hist[:,np.newaxis])
    return model.predict(basis), hist, knots

def knots_from_marginal(marginal, nr_knots, spline_order):
    """
    Determines knot placement based on a marginal distribution.  

    It places knots such that each knot covers the same amount 
    of probability mass. Two of the knots are reserved for the
    borders which are treated seperatly. For example, a uniform
    distribution with 5 knots will cause the knots to be equally 
    spaced with 25% of the probability mass between each two 
    knots.

    Input:
        marginal: Array
            Estimate of the marginal distribution used to estimate
            knot placement.
        nr_knots: int
            Number of knots to be placed.
        spline_order: int 
            Order of the splines

    Returns:
        knots: Array
            Sequence of knot positions
    """
    cumsum = np.cumsum(marginal)
    cumsum = cumsum/cumsum.max()
    borders = np.linspace(0,1,nr_knots)
    knot_placement = [0] + [np.where(cumsum>=b)[0][0] for b in borders[1:-1]] +[len(marginal)-1]
    knots = augknt(knot_placement, spline_order)
    return knots

@Memoize
def spline_base1d(length, nr_knots = 20, spline_order = 5, marginal = None):
    """Computes a 1D spline basis
    
    Input:
        length: int
            length  of each basis
        nr_knots: int
            Number of knots, i.e. number of basis functions.
        spline_order: int
            Order of the splines.
        marginal: array, optional
            Estimate of the marginal distribution of the input to be fitted. 
            If given, it is used to determine the positioning of knots, each 
            knot will cover the same amount of probability mass. If not given,
            knots are equally spaced.
    """
    if marginal is None:
        knots = augknt(np.linspace(0,length, nr_knots), spline_order)
    else:
        knots = knots_from_marginal(marginal, nr_knots, spline_order)
        
    x_eval = np.arange(1,length+1).astype(float)
    Bsplines    = spcol(x_eval,knots,spline_order)
    return Bsplines, knots

@Memoize
def spline_base2d(width, height, nr_knots_x = 20.0, nr_knots_y = 20.0, 
        spline_order = 5, marginal_x = None, marginal_y = None):
    """Computes a set of 2D spline basis functions. 
    
    The basis functions cover the entire space in height*width and can 
    for example be used to create fixation density maps. 

    Input:
        width: int
            width  of each basis
        height: int 
            height of each basis
        nr_knots_x: int
            of knots in x (width) direction.
        nr_knots_y: int
            of knots in y (height) direction.
        spline_order: int
            Order of the spline.
        marginal_x: array, optional
            Estimate of marginal distribution of the input to be fitted
            along the x-direction (width). If given, it is used to determine 
            the positioning of knots, each knot will cover the same amount 
            of probability mass. If not given, knots are equally spaced.
        marginal_y: array, optional
            Marginal distribution along the y-direction (height). If
            given, it is used to determine the positioning of knots.
            Each knot will cover the same amount of probability mass.
    Output:
        basis: Matrix 
            Matrix of size n*(width*height) that contains in each row
            one vectorized basis. 
        knots: Tuple 
            (x,y) are knot arrays that show the placement of knots.
    """
    if not (nr_knots_x<width and nr_knots_y<height):
        raise RuntimeError("Too many knots for size of the base")
    if marginal_x is None:
        knots_x         = augknt(np.linspace(0,width,nr_knots_x), spline_order)
    else:
        knots_x = knots_from_marginal(marginal_x, nr_knots_x, spline_order) 
    if marginal_y is None:
        knots_y         = augknt(np.linspace(0,height, nr_knots_y), spline_order)
    else:
        knots_y = knots_from_marginal(marginal_y, nr_knots_y, spline_order)
    x_eval = np.arange(1,width+1).astype(float)
    y_eval = np.arange(1,height+1).astype(float)    
    spline_setx = spcol(x_eval, knots_x, spline_order)
    spline_sety = spcol(y_eval, knots_y, spline_order)
    nr_coeff = [spline_sety.shape[1], spline_setx.shape[1]]
    dim_bspline = [nr_coeff[0]*nr_coeff[1], len(x_eval)*len(y_eval)]
    # construct 2D B-splines 
    nr_basis = 0
    bspline = np.zeros(dim_bspline)
    for IDX1 in range(0,nr_coeff[0]):
        for IDX2 in range(0, nr_coeff[1]):
            rand_coeff  = np.zeros((nr_coeff[0] , nr_coeff[1]))
            rand_coeff[IDX1,IDX2] = 1
            tmp = np.dot(spline_sety,rand_coeff)
            bspline[nr_basis,:] = np.dot(tmp,spline_setx.T).reshape((1,-1))
            nr_basis = nr_basis+1
    return bspline, (knots_x, knots_y)

def spline_base3d( width, height, depth, nr_knots_x = 10.0, nr_knots_y = 10.0,
        nr_knots_z=10, spline_order = 3, marginal_x = None, marginal_y = None, 
        marginal_z = None):
    """Computes a set of 3D spline basis functions. 
    
    For a description of the parameters see spline_base2d.
    """  
    if not nr_knots_z < depth:
        raise RuntimeError("Too many knots for size of the base")
    basis2d, (knots_x, knots_y) = spline_base2d(height, width, nr_knots_x, 
            nr_knots_y, spline_order, marginal_x, marginal_y)
    if marginal_z is not None:
        knots_z = knots_from_marginal(marginal_z, nr_knots_z, spline_order)
    else:
        knots_z = augknt(np.linspace(0,depth, nr_knots_z), spline_order)
    z_eval = np.arange(1,depth+1).astype(float)
    spline_setz = spcol(z_eval, knots_z, spline_order)
    bspline = np.zeros((basis2d.shape[0]*len(z_eval), height*width*depth))
    basis_nr = 0
    for spline_a in spline_setz.T:
        for spline_b in basis2d:
            spline_b = spline_b.reshape((height, width))
            bspline[basis_nr, :] = (spline_b[:,:,np.newaxis] * spline_a[:]).flat
            basis_nr +=1
    return bspline, (knots_x, knots_y, knots_z)

def spline(x,knots,p,i=0.0):
    """Evaluates the ith spline basis given by knots on points in x"""
    assert(p+1<len(knots))
    return np.array([N(float(u),float(i),float(p),knots) for u in x])

def spcol(x,knots,spline_order):
    """Computes the spline colocation matrix for knots in x.
    
    The spline collocation matrix contains all m-p-1 bases 
    defined by knots. Specifically it contains the ith basis
    in the ith column.
    
    Input:
        x: vector to evaluate the bases on
        knots: vector of knots 
        spline_order: order of the spline
    Output:
        colmat: m x m-p matrix
            The colocation matrix has size m x m-p where m 
            denotes the number of points the basis is evaluated
            on and p is the spline order. The colums contain 
            the ith basis of knots evaluated on x.
    """
    colmat = np.nan*np.ones((len(x),len(knots) - spline_order-1))
    for i in range(0,len(knots) - spline_order -1):
        colmat[:,i] = spline(x,knots,spline_order,i)
    return colmat
    
def augknt(knots,order):
    """Augment knot sequence such that some boundary conditions 
    are met."""
    a = []
    [a.append(knots[0]) for t in range(0,order)]
    [a.append(k) for k in knots]
    [a.append(knots[-1]) for t in range(0,order)]
    return np.array(a)     

def N(u,i,p,knots):
    """Compute Spline Basis
    
    Evaluates the spline basis of order p defined by knots 
    at knot i and point u.
    """
    if p == 0:
        if knots[i] < u and u <=knots[i+1]:
            return 1.0
        else:
            return 0.0
    else:
        try:
            k = (( float((u-knots[i]))/float((knots[i+p] - knots[i]) )) 
                    * N(u,i,p-1,knots))
        except ZeroDivisionError:
            k = 0.0
        try:
            q = (( float((knots[i+p+1] - u))/float((knots[i+p+1] - knots[i+1])))
                    * N(u,i+1,p-1,knots))
        except ZeroDivisionError:
            q  = 0.0 
        return float(k + q)

 
