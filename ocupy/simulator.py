
"""This module implements a generator of data 
with given second-order dependencies"""

from math import radians, ceil, floor, cos, sin 
import random
import ocupy
from ocupy import fixmat
import spline_base
import numpy as np
from progressbar import ProgressBar, Percentage, Bar
import functools  
import simulator

class memoize(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned 
    (not reevaluated).
    '''
      
    def __init__(self, func):
        self.func = func
        self.cache = {}
        
    def __call__(self,x, cumsum, r):
        try:
            return self.cache[cumsum][r]
        except KeyError:
            value = self.func(x,cumsum, r)
            try:
                self.cache[cumsum][r] = value
            except KeyError:
                self.cache[cumsum] = {r:value}
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(cumsum,r)
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):        
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)
        
class AbstractSim(object):
    """
    Abstract Object for Simulator creation
    """
    def __init__(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def parameters(self):
        raise NotImplementedError

class FixGen(AbstractSim):
    """
    Generates fixation data.

    The FixGen object creates a representation of the second order dependence structures
    between saccades contained in the fixmat given as input. It is then able to generate 
    and return a fixmat which replicates these dependencies, while consisting of different 
    fixations.
    In order to work, the initialized FixGen obejct has to initialize its data by calling
    the method initializeData():

            >>> gen = simulator.FixGen(fm)
            >>> gen.initializeData()
    

    Separating this time-consuming step from the initialization is helpful in cases of 
    parallelization.
    Data is generated upon calling the method sample_many(num_samples = X).  
    """
    def __init__(self, fm):
        """
        Creates a new FixGen object upon a certain fixmat
        
        Parameters: 
            fm: ocupy.fixmat
                The fixation data to replicate in fixmat format.
        """
        if type(fm)==ocupy.fixmat.FixMat:
            self.fm = fm
        else:
            raise TypeError("Not a valid argument, insert fixmat")

        self.nosamples = []
        
    def initializeData(self, fit = None, full_H1=None, max_length = 40, in_deg = True):
        """
        Prepares the data to be replicated. Calculates the second-order length 
        and angle dependencies between saccades and stores them in a fitted 
        histogram.
        
        Parameters:
            fit : function, optional
                The method to use for fitting the histogram
            full_H1 : twodimensional numpy.ndarray, optional
                Where applicable, the distribution of angle and length
                differences to replicate with dimensions [73,361]
        """
        a, l, ad, ld = anglendiff(self.fm, roll=1, return_abs = True)
        if in_deg:
            self.fm.pixels_per_degree = 1
            
        samples = np.zeros([3, len(l[0])])
        samples[0] = l[0]/self.fm.pixels_per_degree
        samples[1] = np.roll(l[0]/self.fm.pixels_per_degree,-1)
        samples[2] = np.roll(reshift(ad[0]),-1)
        z = np.any(np.isnan(samples), axis=0)
        samples = samples[:,~np.isnan(samples).any(0)]
           
        if full_H1 is None:   
            self.full_H1 = []
            for i in range(1, int(ceil(max_length+1))):
                idx = np.logical_and(samples[0]<=i, samples[0]>i-1)
                if idx.any():
                    self.full_H1.append(makeHist(samples[2][idx], samples[1][idx], fit=fit, 
                                                bins=[np.linspace(0,max_length-1,max_length),np.linspace(-180,180,361)]))
                    # Sometimes if there's only one sample present there seems to occur a problem
                    # with histogram calculation and the hist is filled with nans. In this case, dismiss
                    # the hist.
                    if np.isnan(self.full_H1[-1]).any():
                        self.full_H1[-1] = np.array([])
                    self.nosamples.append(len(samples[2][idx]))
                else:
                    self.full_H1.append(np.array([]))
                    self.nosamples.append(0)
        else:
            self.full_H1 = full_H1
                
        self.firstLenAng_cumsum, self.firstLenAng_shape = (
                                        compute_cumsum(firstSacDist(self.fm)))
        self.probability_cumsum = []
       
        for i in range(len(self.full_H1)):
            if self.full_H1[i] == []:
                self.probability_cumsum.append(np.array([]))
            else:
                self.probability_cumsum.append(np.cumsum(self.full_H1[i].flat))
               
        self.trajLen_cumsum, self.trajLen_borders = trajLenDist(self.fm)
        
        min_distance = 1/np.array([min((np.unique(self.probability_cumsum[i]) \
                        -np.roll(np.unique(self.probability_cumsum[i]),1))[1:]) \
                        for i in range(len(self.probability_cumsum))])
        # Set a minimal resolution
        min_distance[min_distance<10] = 10

        self.linind = {}
        for i in range(len(self.probability_cumsum)):
            self.linind['self.probability_cumsum '+repr(i)] = np.linspace(0,1,min_distance[i])[0:-1]
        
        for elem in ['self.firstLenAng_cumsum', 'self.trajLen_cumsum']:
            self.linind[elem] = np.linspace(0, 1, 1/min((np.unique(eval(elem))-np.roll(np.unique(eval(elem)),1))[1:]))[0:-1]
        
    def _calc_xy(self, (x, y), angle, length):
        """
        Calculates the coordinates after a specific saccade was made.
        
        Parameters:
            (x,y) : tuple of floats or ints
                The coordinates before the saccade was made
            angle : float or int
                The angle that the next saccade encloses with the 
                horizontal display border
            length: float or int
                The length of the next saccade
        """
        return (x+(cos(radians(angle))*length),
                y+(sin(radians(angle))*length))
                
    def _draw(self, prev_angle = None, prev_length = None):
        """
        Draws a new length- and angle-difference pair and calculates
        length and angle absolutes matching the last saccade drawn.

        Parameters:
            prev_angle : float, optional
                The last angle that was drawn in the current trajectory
            prev_length : float, optional
                The last length that was drawn in the current trajectory
            
            Note: Either both prev_angle and prev_length have to be given 
            or none; if only one parameter is given, it will be neglected.
        """
        
        if (prev_angle is None) or (prev_length is None):
            (length, angle)= np.unravel_index(self.drawFrom('self.firstLenAng_cumsum', self.getrand('self.firstLenAng_cumsum')),
                                                self.firstLenAng_shape)
            angle = angle-((self.firstLenAng_shape[1]-1)/2) 
            angle += 0.5
            length += 0.5
            length *= self.fm.pixels_per_degree
        else:
            ind = int(floor(prev_length/self.fm.pixels_per_degree))
            while ind >= len(self.probability_cumsum):
                ind -= 1

            while not(self.probability_cumsum[ind]).any():
                ind -= 1
                
            J, I = np.unravel_index(self.drawFrom('self.probability_cumsum '+repr(ind),self.getrand('self.probability_cumsum '+repr(ind))), 
                                    self.full_H1[ind].shape)
            angle = reshift((I-self.full_H1[ind].shape[1]/2) + prev_angle)
            angle += 0.5
            length = J+0.5
            length *= self.fm.pixels_per_degree
        return angle, length
    
    def parameters(self):
        return {'fixmat':self.fm, 'sampling_dist':self.full_H1}

    def generate(self, num_samples = 10000, multiproc = False):
        if multiproc:
            from multiprocessing import Pool, cpu_count
            try:
                cores = cpu_count()
            except NotImplementedError:
                cores = 4
                
            arguments = []
            for i in range(cores):
                arguments.append((simulator.FixGen(self.fm),num_samples/cores))
                
            pool = Pool(processes = cores)
            fms = pool.map(multiprocess, arguments)
            out = fms[0]
            out.SUBJECTINDEX = [0]*len(out.x) # Delete SUBJECTINDEX-stuff after update of ocupy.
            for i in range(1,len(fms)):
                fms[i].SUBJECTINDEX = [i]*len(fms[i].x)
                out.join(fms[i])
            return out
            
        else: 
            return self.sample_many(num_samples = num_samples)
            
    def sample_many(self, num_samples = 2000):
        """
        Generates a given number of trajectories, using the method sample(). 
        Returns a fixmat with the generated data.
        
        Parameters:
            num_samples : int, optional
                The number of trajectories that shall be generated.
        """     
        x = []
        y = []
        fix = []
        sample = []
        
        # XXX: Delete ProgressBar
        pbar = ProgressBar(widgets=[Percentage(),Bar()], maxval=num_samples).start()
        
        for s in xrange(0, num_samples):
            for i, (xs, ys) in enumerate(self.sample()):
                x.append(xs)
                y.append(ys)
                fix.append(i+1)
                sample.append(s)
            pbar.update(s+1)
            
        fields = {'fix':np.array(fix), 'y':np.array(y), 'x':np.array(x)}
        param = {'pixels_per_degree':self.fm.pixels_per_degree}
        out =  fixmat.VectorFixmatFactory(fields, param)
        return out
    
    def sample(self):
        """
        Draws a trajectory length, first coordinates, lengths, angles and 
        length-angle-difference pairs according to the empirical distribution. 
        Each call creates one complete trajectory.
        """
        lenghts = []
        angles = []
        coordinates = []
        fix = []
        sample_size = int(round(self.trajLen_borders[self.drawFrom('self.trajLen_cumsum', self.getrand('self.trajLen_cumsum'))]))

        coordinates.append([0, 0])
        fix.append(1)
        
        while len(coordinates) < sample_size:
            if len(lenghts) == 0 and len(angles) == 0:          
                angle, length = self._draw(self)
            else:
                angle, length = self._draw(prev_angle = angles[-1], 
                                            prev_length = lenghts[-1])  
                        
            x, y = self._calc_xy(coordinates[-1], angle, length) 
            
            coordinates.append([x, y])
            lenghts.append(length) 
            angles.append(angle)
            fix.append(fix[-1]+1)
        return coordinates

    def getrand(self, name):
        return random.choice(self.linind[name])
 
    @memoize    
    def drawFrom(self, cumsum, r):
        """
        Draws a value from a cumulative sum.
        
        Parameters: 
            cumsum : array
                Cumulative sum from which shall be drawn.

        Returns:
            int : Index of the cumulative sum element drawn.
        """
        a = cumsum.rsplit()
        if len(a)>1:
            b = eval(a[0])[int(a[1])]
        else:
            b = eval(a[0])
            
        return np.nonzero(b>=r)[0][0]
        
def multiprocess(arguments):
    (sim, numsamples) = arguments
    sim.initializeData()
    return sim.sample_many(num_samples = numsamples)

def makeAngLenHist(ad, ld, fm = None, collapse=True, fit=spline_base.fit2d):
    """
    Histograms and performs a spline fit on the given data, 
    usually angle and length differences.
    
    Parameters:
        ad : array
            The data to be histogrammed along the x-axis. 
            May range from -180 to 180.
        ld : array
            The data to be histogrammed along the y-axis.
            May range from -36 to 36.
        collapse : boolean
            If true, the histogrammed data will include 
            negative values on the x-axis. Else, the histogram
            will be collapsed along x = 0, and thus contain only 
            positive angle differences
        fit : function or None, optional
            The function to use in order to fit the data. 
            If no fit should be applied, set to None
        fm  : fixmat or None, optional
            If given, the angle and length differences are calculated
            from the fixmat and the previous parameters are overwritten.
    """
    
    if fm:
        ad,ld = anglendiff(fm, roll=2)
        ad, ld = ad[0], ld[0]
        
    ld = ld[~np.isnan(ld)]
    ad = reshift(ad[~np.isnan(ad)])

    if collapse:
        e_y = np.linspace(-36.5, 36.5, 74)
        e_x = np.linspace(0, 180, 181)
        H = makeHist(abs(ad), ld, fit=fit, bins=[e_y, e_x])

        H = H/H.sum()
        
        return H
    else:
        e_x = np.linspace(-180, 180, 361)
        e_y = np.linspace(-36.5, 36.5, 74)
        return makeHist(ad, ld, fit=fit, bins=[e_y, e_x])

def makeHist(x_val, y_val, fit=spline_base.fit2d, 
            bins=[np.linspace(-36.5,36.5,74),np.linspace(-180,180,361)]):
    """
    Constructs a (fitted) histogram of the given data.
    
    Parameters:
        x_val : array
            The data to be histogrammed along the x-axis. 
        y_val : array
            The data to be histogrammed along the y-axis.
        fit : function or None, optional
            The function to use in order to fit the data. 
            If no fit should be applied, set to None
        bins : touple of arrays, giving the bin edges to be 
            used in the histogram. (First value: y-axis, Second value: x-axis)
    """
    
    y_val = y_val[~np.isnan(y_val)]
    x_val = x_val[~np.isnan(x_val)]
    
    samples = zip(y_val, x_val)
    K, xedges, yedges = np.histogram2d(y_val, x_val, bins=bins)

    if (fit is None):
        return K/ K.sum()
   
    # Check if given attr is a function
    elif hasattr(fit, '__call__'):
        H = fit(np.array(samples), bins[0], bins[1], p_est=K)[0]
        return H/H.sum()
    else:
        raise TypeError("Not a valid argument, insert spline function or None")
        
def shuffled_anglendiff(angles, lengths, roll = 2, return_abs = False):
    sangle_diffs = []
    slength_diffs = []
    sangles = []
    slengths = []
    
    index = np.random.permutation(len(angles))
    sangles.append(angles[np.array(index)])
    slengths.append(lengths[np.array(index)])
    
    sangle_diffs.append(sangles[0] - np.roll(sangles[0],1))
    slength_diffs.append(slengths[0] - np.roll(slengths[0],1))
    
    # Restore height and width information from shuffled saccades using the law
    # a/sin(alpha)=b/sin(beta). Note: All angles have to be converted back to radians.
    tmp_sheights = np.array(slengths[0]) * np.sin(np.radians(sangles[0])) / sin(np.radians(90))
    sheights = tmp_sheights[:]
    # ... and using the law of tangens
    tmp_swidths =  sheights / np.tan(np.radians(sangles[0]))
    swidths = tmp_swidths[:]
    
    for r in range(2,roll+1):
        tmp_swidths = tmp_swidths + np.roll(swidths,r-1)
        tmp_sheights = tmp_sheights + np.roll(sheights, r-1)
        sangles.append(np.degrees(np.arctan2(tmp_sheights,tmp_swidths)))
        slengths.append((tmp_swidths**2+tmp_sheights**2)**.5)
        
        sangle_diffs.append(sangles[0] - np.roll(sangles[-1],1))
        slength_diffs.append(slengths[0] - np.roll(slengths[-1],1)) 
    
    if return_abs:
        return sangles, slengths, sangle_diffs, slength_diffs
    else:
        return sangle_diffs, slength_diffs
            
def anglendiff(fm, roll = 2, return_abs=False):
    """
    Calculates the lengths and angles of the saccades contained in the fixmat
    as well as length- and angle differences between consecutive saccades.
    Returns a nested vector structure that gives these multi-order differences
    in the following order:
        
        >>> anglendiff(fm, roll = 2)
        Out: [[AngleDiffs 2nd order], [AngleDiffs 3rd order]], 
             [[LengthDiffs 2nd order], [LengthDiffs 3rd order]]
        
    
    Parameters: 
        fm : ocupy.fixmat object 
            The fixmat with the data that shall be analyzed.
        roll : int, optional
            The maximum order of the dependence structure that shall be 
            analyzed -1.
            
                >>> anglendiff(fm, roll=2)   # Analyzes the data up to 3rd order
            
            If none is given, only the second order properties are calculated.
        return_abs : boolean, optional
            By default, the method returns only length-angle difference pairs. 
            If return_abs is set to true, the length and angle absolutes are 
            returned as well.
            
                >>> angles, lengths, angle_diffs, length_diffs = 
                            anglendiff(fm, return_abs = True)
    """
    
    angle_diffs = []
    length_diffs = []
    lengths = []
    angles  = []
    
    for r in range(1, roll+1):
        heights = (fm.y - np.roll(fm.y, r)).astype(float)
        widths = (fm.x - np.roll(fm.x, r)).astype(float)
        heights[fm.fix <= min(fm.fix)+r-1]=np.nan
        widths[fm.fix <= min(fm.fix)+r-1]=np.nan
        
        lengths.append((widths**2+heights**2)**.5)
        angles.append(np.degrees(np.arctan2(heights, widths)))
        
        length_diffs.append(lengths[0] - np.roll(lengths[r-1], 1))
        
        # -360: straight saccades, -180: return saccades, 0: straight saccades,
        # 180: return saccades, 360: no return saccades
        angle_diffs.append(angles[0] - np.roll(angles[r-1], 1))
                
    if return_abs == True:
        return angles, lengths, angle_diffs, length_diffs
        
    else:
        return angle_diffs, length_diffs
 
def compute_cumsum(H):
    """
        Computes the cumulative sum of a given 2D distribution
        
        Parameters:
            H : twodimensional numpy.ndarray
    
    """  
    return np.cumsum(H.flat), H.shape
    
def firstSacDist(fm):
    """
        Computes the distribution of angle and length
        combinations that were made as first saccades
        
        Parameters:
            fm : ocupy.fixmat 
                The fixation data to be analysed
    
    """  
    ang, leng, ad, ld = anglendiff(fm, return_abs=True)
    y_arg = leng[0][np.roll(fm.fix == min(fm.fix), 1)]/fm.pixels_per_degree
    x_arg = reshift(ang[0][np.roll(fm.fix == min(fm.fix), 1)])
    bins = [range(int(ceil(np.nanmax(y_arg)))+1), np.linspace(-180, 180, 361)]
    return makeHist(x_arg, y_arg, fit=None, bins = bins)
    
def trajLenDist(fm):
    """
        Computes the distribution of trajectory lengths, i.e.
        the number of saccades that were made as a part of one trajectory
        
        Parameters:
            fm : ocupy.fixmat 
                The fixation data to be analysed
    
    """  
    trajLen = np.roll(fm.fix, 1)[fm.fix == min(fm.fix)]
    val, borders = np.histogram(trajLen, 
                    bins=np.linspace(-0.5, max(trajLen)+0.5, max(trajLen)+2))
    cumsum = np.cumsum(val.astype(float) / val.sum())
    return cumsum, borders

def reshift(I):
    """
    Transforms the given number element into a range of [-180, 180],
    which covers all possible angle differences. This method reshifts larger or 
    smaller numbers that might be the output of other angular calculations
    into that range by adding or subtracting 360, respectively. 
    To make sure that angular data ranges between -180 and 180 in order to be
    properly histogrammed, apply this method first.
    
    
    Parameters: 
        I : array or list or int or float
            Number or numbers that shall be reshifted.
    
    Returns:
        numpy.ndarray : Reshifted number or numbers as array
    """
    # Output -180 to +180
    if type(I)==list:
        I = np.array(I)
    
    return ((I-180)%360)-180
