import numpy as np
import scikits.statsmodels.api as sm

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

def spline_pdf(samples,e_y,e_x,base=None,nr_knots_x=3,nr_knots_y=3,hist=None):
    height = len(e_y)-1
    width = len(e_x)-1
    if not base:
        base = spline_base(height, width, nr_knots_x,nr_knots_y , 3)
    if hist == None:
        (hist, _) = np.histogramdd(samples, (e_y, e_x))
    poiss_model = sm.Poisson(hist.reshape(-1,1), base.T)
    results = poiss_model.fit(maxiter=100, method='bfgs')
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

def spline_base(height, width, Nr_Knots_x = 5.0, Nr_Knots_y = 5.0, 
        spline_order = 3,scale_factor=None):
    """Computes a set of 2D spline basis functions. 
    
    The basis functions cover the entire space in height*width and can 
    for example be used to create fixation density maps. 

    Input:
        height: int
            height of each basis
        width: int 
            widht of each basis
        Nr_Knots_x: int
            # of knots in x direction. These will be equally spaced
        Nr_Knots_y: int
            # of knots in y direction. These will be equally spaced
        spline_order: int
            Order of the spline.
    Output:
        bases: Matrix of size n*(width*height) that contains in each row
            one vectorized basis.
    """
    Knots_x         = augknt(np.linspace(0,width,Nr_Knots_x),spline_order)
    Knots_y         = augknt(np.linspace(0,height,Nr_Knots_y),spline_order)
    if not scale_factor:
        x_eval = np.arange(1,width+1).astype(float)
        y_eval = np.arange(1,height+1).astype(float)
    else:
        x_eval = np.linspace(1,width, width*scale_factor)
        y_eval = np.linspace(1,height, height*scale_factor)
    spline_setx     = spcol(x_eval,Knots_x,spline_order)
    spline_sety     = spcol(y_eval,Knots_y,spline_order)
    Nr_coeff        = [spline_sety.shape[1], spline_setx.shape[1]]
    DIM_Bspline     = [Nr_coeff[0]*Nr_coeff[1], len(x_eval)*len(y_eval)]
    # construct 2D B-splines 
    Nr_basis        = 0
    Bspline         = np.zeros((DIM_Bspline[0],DIM_Bspline[1]))
    for IDX1 in range(0,Nr_coeff[0]):
        for IDX2 in range(0, Nr_coeff[1]):
            Rand_coeff      = np.zeros((Nr_coeff[0],Nr_coeff[1]))
            Rand_coeff[IDX1,IDX2] = 1
            tmp = np.dot(spline_sety,Rand_coeff)
            Bspline[Nr_basis,:]     = np.dot(tmp,spline_setx.T).reshape((1,-1))
            Nr_basis                = Nr_basis+1
    return Bspline


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

 
