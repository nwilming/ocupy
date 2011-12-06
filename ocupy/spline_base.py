import numpy as np
import scikits.statsmodels as sm
from scikits.learn import linear_model

def spline_fdm(fm,base_spec = None, spline_order = 3):
    # Set up parameters in relation to image_size
    if not base_spec:
        height = np.round(fm.image_size[0] / (.5*fm.pixels_per_degree))
        width = np.round(fm.image_size[1] /(.5*fm.pixels_per_degree))
        nr_knots_x = 4#width/4
        nr_knots_y = 4#height/4
    else:
        height, width, nr_knots_y, nr_knots_x = base_spec
    # compute down scale factor
    down_scale = float(height)/fm.image_size[0] 
    # compute basis
    #base = spline_base(height, width, nr_knots_x, nr_knots_y, spline_order)
    # compute target histogram
    # this specifies left edges of the histogram bins, i.e. fixations between
    # ]0 binedge[0]] are included. --> fixations are ceiled
    e_y = np.arange(0, height+1)
    e_x = np.arange(0, width+1)
    samples = np.array(zip((down_scale*fm.y), (down_scale*fm.x)))
    return spline_pdf(samples,e_y,e_x)

#@deprecated
def spline_pdf(samples,e_y,e_x,base=None,nr_knots_x=3,nr_knots_y=3,hist=None):
    height = len(e_y)-1
    width = len(e_x)-1
    if not base:
        base = spline_base2d(height, width, nr_knots_x,nr_knots_y , 3)
    if hist == None:
        (hist, _) = np.histogramdd(samples, (e_y, e_x))
    poiss_model = sm.Poisson(hist.reshape(-1,1), base.T)
    results = poiss_model.fit(maxiter=500, method='bfgs')
    big_base = spline_base(height*1,width,
          nr_knots_x, nr_knots_y, 3)
    #big_base = sm.add_constant(big_base.T)
    return np.exp(np.dot(big_base.T,results.params)).reshape((1*height, width))
   
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
    are met. However, right now I have not really a clue why this
    is needed."""
    a = []
    [a.append(knots[0]) for t in range(0,order)]
    [a.append(k) for k in knots]
    [a.append(knots[-1]) for t in range(0,order)]
    return np.array(a)     

def fit3d(samples, e_y, e_x, e_z, remove_zeros = False, **kw):
    height, width, depth = len(e_y)-1, len(e_x)-1, len(e_z)-1 
    
    (p_est, _) = np.histogramdd(samples, (e_y, e_x, e_z))
    p_est = p_est/sum(p_est.flat)
    p_est = p_est.flatten()
    if remove_zeros:
        non_zero = ~(p_est == 0)
    else:
        non_zero = (p_est >= 0)
    basis = spline_base3d(height, width, depth, **kw)
    model = linear_model.BayesianRidge()
    model.fit(basis[:, non_zero].T, p_est[:,np.newaxis][non_zero,:])
    return model.predict(basis.T).reshape((height, width, depth)), p_est.reshape((height,width, depth))
       
       
def fit2d(samples,e_y,e_x, remove_zeros = False, **kw):
    height = len(e_y)-1
    width = len(e_x)-1   
    (p_est, _) = np.histogramdd(samples, (e_y, e_x))
    p_est = p_est/sum(p_est.flat)
    mx =  sum(p_est,1)
    my = sum(p_est,0)
    p_est = p_est.flatten()
    if remove_zeros:
        non_zero = ~(p_est == 0)
    else:
        non_zero = (p_est >= 0)
    basis = spline_base2d(height, width, marginal_x = mx, marginal_y = my, **kw)
    model = linear_model.BayesianRidge()
    model.fit(basis[:, non_zero].T, p_est[:,np.newaxis][non_zero,:])
    return model.predict(basis.T).reshape((height, width)), p_est.reshape((height,width))

def fit1d(samples, e, bayesian_ridge = False):
    samples = samples[~np.isnan(samples)]
    length = len(e)-1
    hist,_ = np.histogramdd(samples, (e,))
    hist = hist/sum(hist)
    basis = spline_base1d(length, nr_knots = 20, marginal = hist, spline_order = 5)
    non_zero = hist>0
    model = linear_model.BayesianRidge()
    model.fit(basis[non_zero, :], hist[:,np.newaxis][non_zero,:])
    return model.predict(basis), hist

def find_fit_parameters2d(samples, e_x, e_y, max_params = 50):
    x,y = np.mgrid[1:max_params, 1:max_params]
    idx = (x**2+y**2)**.5 < max_params
    x,y = x[idx], y[idx]    
    results = []
    for xp, yp in zip(x,y):
        shuffle_idx = np.arange(0,len(samples[0]))
        score = []
        for r in range(10):
            np.random.shuffle(shuffle_idx)
            train_samples = (samples[0][shuffle_idx][::2], samples[1][shuffle_idx][0::2])
            test_samples = (samples[0][shuffle_idx][1::2], samples[1][shuffle_idx][1::2])
            fit, hist = fit2d(train_samples, e_y, e_x, nr_knots_x=xp, nr_knots_y=yp)
            idx = [np.where((e_x > x) & (x >= e_x)) for x in samples[0]]
            idy = [np.where((e_y > y) & (y >= e_y)) for y in samples[1]]
            score += np.prod(fit[idx, idy])
        results += [(xp, yp, np.mean(score))]
    return results

def knots_from_marginal(marginal, nr_knots, spline_order):
    cumsum = np.cumsum(marginal)
    cumsum = cumsum/cumsum.max()
    borders = np.linspace(0,1,nr_knots)[1:]
    intervals = [0] + [np.where(cumsum>=b)[0][0] for b in borders]
    knot_placement = [t1+((t2-t1)/2) for t1,t2 in zip(intervals[:-1], intervals[1:])]
    if not 0 in knot_placement:
        knot_placement = [0] + knot_placement
    if not len(marginal) in knot_placement:
        knot_placement += [len(marginal)]
    knots = augknt(knot_placement, spline_order)
    return knots
    
def spline_base1d(length, nr_knots = 5, spline_order = 20, marginal = None):
    """Computes a 1D spline basis"""
    if marginal is None:
        knots = augknt(np.linspace(0,length, nr_knots), spline_order)
    else:
        knots = knots_from_marginal(marginal, nr_knots, spline_order)
        
    x_eval = np.arange(1,length+1).astype(float)
    Bsplines    = spcol(x_eval,knots,spline_order)
    return Bsplines

def spline_base2d(height, width, nr_knots_x = 10.0, nr_knots_y = 10.0, 
        spline_order = 3, marginal_x = None, marginal_y = None):
    """Computes a set of 2D spline basis functions. 
    
    The basis functions cover the entire space in height*width and can 
    for example be used to create fixation density maps. 

    Input:
        height: int
            height of each basis
        width: int 
            widht of each basis
        nr_Knots_x: int
            # of knots in x direction. These will be equally spaced
        nr_Knots_y: int
            # of knots in y direction. These will be equally spaced
        spline_order: int
            Order of the spline.
    Output:
        bases: Matrix of size n*(width*height) that contains in each row
            one vectorized basis.
    """
    assert nr_knots_x<width and nr_knots_y<height, "Too many knots for size of the base"
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
    DIM_Bspline = [nr_coeff[0]*nr_coeff[1], len(x_eval)*len(y_eval)]
    # construct 2D B-splines 
    nr_basis = 0
    bspline = np.zeros((DIM_Bspline[0],DIM_Bspline[1]))
    for IDX1 in range(0,nr_coeff[0]):
        for IDX2 in range(0, nr_coeff[1]):
            rand_coeff  = np.zeros((nr_coeff[0] , nr_coeff[1]))
            rand_coeff[IDX1,IDX2] = 1
            tmp = np.dot(spline_sety,rand_coeff)
            bspline[nr_basis,:] = np.dot(tmp,spline_setx.T).reshape((1,-1))
            nr_basis = nr_basis+1
    return bspline

def spline_base3d(height, width, depth, nr_knots_x = 10.0, nr_knots_y = 10.0,
        nr_knots_z=10, spline_order = 3, marginal_x = None, marginal_y = None):
    """Computes a set of 3D spline basis functions. 
    """    
    basis2d = spline_base2d(height, width, nr_knots_x, nr_knots_y, spline_order, marginal_x, marginal_y)
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
    return bspline


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

 
