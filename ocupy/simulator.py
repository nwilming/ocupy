
"""This module implements a generator of data 
with given second-order dependencies"""

from math import radians, ceil, floor, cos, sin 
import random
import ocupy
from ocupy import fixmat
import spline_base
import numpy as np
import pdb
import cPickle
from progressbar import ProgressBar, Percentage, Bar


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

def makeAngLenHist(ad, ld, collapse=True, fit=spline_base.fit2d):
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
            will be folded along x = 0, and thus contain only 
            positive elements
        fit : function or None, optional
            The function to use in order to fit the data. 
            If no fit should be applied, set to None
    """

    ld = ld[~np.isnan(ld)]
    ad = reshift(ad[~np.isnan(ad)]) 
    if collapse:
        e_y = np.linspace(-36.5, 36.5, 74)
        e_x = np.linspace(0, 180, 181)
        H = makeHist(abs(ad), ld, fit=fit, bins=[e_y, e_x])
        '''
        H[:,0]*=2  
        H[:,-1]*=2
        '''
        return H
    else:
        e_x = np.linspace(-180.5, 179.5, 361)
        e_y = np.linspace(-36.5, 36.5, 74)
        ad[ad > 179.5] -= 360
        return makeHist(ad, ld, fit=fit, bins=[e_y, e_x])

def makeHist(x_val, y_val, fit=spline_base.fit2d, 
            bins=[np.linspace(-36.5,36.5,74),np.linspace(-180.5,180.5,362)]):
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
    
    K, xedges, yedges = np.histogram2d(y_val, x_val, bins=bins)
    K = K / K.sum()

    if (fit is None):
        return K
    
    # Check if given attr is a function
    elif hasattr(fit, '__call__'):
        samples = zip(y_val, x_val)
        H = fit(np.array(samples), bins[0], bins[1], p_est=K)[0]
        return H/H.sum()
        
    else:
        raise TypeError("Not a valid argument, insert spline function or None")


class FixGen(AbstractSim):
    """
    Generates fixation data.
    The FixGen object creates a representation of the second order dependence 
    structures between saccades contained in the fixmat given as input. It is 
    then able to generate and return a fixmat which replicates these 
    dependencies, while consisting of different fixations.
    In order to work, the initialized FixGen obejct has to initialize its data 
    by calling the method initialize_data():

            >>> gen = simulator.FixGen(fm)
            >>> gen.initialize_data()
    
    Separating this time-consuming step from the initialization is helpful in 
    cases of parallelization.
    Data is generated upon calling the method sample_many(num_samples = 1000).  
    """
    def __init__(self, fm, firstfixcentered=False):
        """
        Creates a new FixGen object upon a certain fixmat
        
        Parameters: 
            fm: ocupy.fixmat
                The fixation data to replicate in fixmat format.
            firstfixcentered: boolean, optional
                If the first fixation was always kept centered in the given 
                fixmat, change this attribute (independent of whether or not 
                this first fixation was actually deleted from the given fixmat)
        """
        if type(fm)==ocupy.fixmat.FixMat:
            self.fm = fm
        else:
            raise TypeError("Not a valid argument, insert fixmat")

        self.firstfixcentered = firstfixcentered
        self.nosamples = []
        self.noDist = 0
                    
        
    def initializeData(self, fit=spline_base.fit2d, full_H1=None):
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
        
        samples = np.zeros([3, len(l[0])])
        samples[0] = l[0]/45
        samples[1] = np.roll(l[0]/45,-1)
        samples[2] = np.roll(reshift(ad[0]),-1)
        z = np.any(np.isnan(samples), axis=0)
        samples = samples[:,~np.isnan(samples).any(0)]
            
        
        if full_H1 is None:   
            self.full_H1 = []
            screen_diag = ((self.fm.image_size[0]**2+self.fm.image_size[1]**2)**.5)/45
            for i in range(1, int(ceil(screen_diag))):
                idx = np.logical_and(samples[0]<=i, samples[0]>i-1)
                if idx.any() == True:
                    self.full_H1.append(makeHist(samples[2][idx], samples[1][idx], fit=fit, 
                                                bins=[np.linspace(-0.5,36.5,38),np.linspace(-180.5,180.5,362)]))
    
                    self.nosamples.append(len(samples[2][idx]))
                else:
                    self.full_H1.append(np.array([]))
                    self.nosamples.append([0])
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
        
        self.firstcoo_cumsum = compute_cumsum(first_fixation_coordinates(
            self.fm))[0]
        self.trajLen_cumsum, self.trajLen_borders = trajectory_lengths(self.fm)
        
        # Counters for saccades that have to be canceled during the process
        self.canceled = 0
        self.minusSaccades = 0
        self.drawnAngDiff = []  # XXX: DELETE UNUSED VARS
        self.drawnAng = []
        self.drawnLen = []
        self.firstLen = []
        self.firstAng = []
        
        
                   
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
            (length, angle)= np.unravel_index(draw_from(self.firstLenAng_cumsum),
                                                self.firstLenAng_shape)
            angle = angle-180  # XXX: ADJUST
            self.firstAng.append(angle) # XXX: DELETE
            #angle = angle-((self.firstLenAng_shape[1]-1)/2) 
            length+=0.5
            length*=self.fm.pixels_per_degree
            self.firstLen.append(length)  # XXX: DELETE
        else:
            ind = int(floor(prev_length/45))
            while ind >= len(self.probability_cumsum):
                ind -= 1
                self.noDist += 1

            while not(self.probability_cumsum[ind]).any():
                ind -= 1
                self.noDist += 1
                
            J, I = np.unravel_index(drawFrom(self.probability_cumsum[ind]), 
                                    self.full_H1[ind].shape)
            angle = reshift((I-180)+prev_angle)  # XXX: Make more general (see line below)
            #angle = reshift((I-self.full_H1[ind].shape[1]/2) + prev_angle)
            self.drawnAngDiff.append(I)
            self.drawnAng.append(angle)
            #import pylab as pp
            #pp.plot(self.probability_cumsum[ind])
            length = J+0.5
            length *= self.fm.pixels_per_degree
            self.drawnLen.append(length)
        return angle, length
    
    def parameters(self):
        return {'fixmat':self.fm, 'sampling_dist':self.full_H1}

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
        param = {'image_size':self.fm.image_size,
                'pixels_per_degree':self.fm.pixels_per_degree}
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
        sample_size = int(round(draw_from(self.trajLen_cumsum, 
                            borders=self.trajLen_borders)))
        
        if (self.firstfixcentered == True):
            coordinates.append([self.fm.image_size[1]/2, 
                self.fm.image_size[0]/2])
        else:
            K, L = (np.unravel_index(draw_from(self.firstcoo_cumsum),
                                [self.fm.image_size[0],self.fm.image_size[1]]))
            coordinates.append([L, K])
        fix.append(1)
        while len(coordinates) < sample_size:
            if len(lenghts) == 0 and len(angles) == 0:          
                angle, length = self._draw(self)
            else:
                angle, length = self._draw(prev_angle = angles[-1], 
                                            prev_length = lenghts[-1])  
                        
            x, y = calc_xy(coordinates[-1], angle, length) 
            
            if (length<0):
                self.minusSaccades += 1 # Drawn saccade length not possible
            else:
                coordinates.append([x, y])
                lenghts.append(length) 
                angles.append(angle)
                fix.append(fix[-1]+1)
        return coordinates


def make_angle_length_hist(ad, ld, collapse=True, fit=spline_base.spline_pdf):
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
            will be folded along x = 0, and thus contain only 
            positive elements
        fit : function or None, optional
            The function to use in order to fit the data. 
            If no fit should be applied, set to None
    """


    ld = ld[~np.isnan(ld)]
    ad = reshift(ad[~np.isnan(ad)]) 

    if collapse:
        e_y = np.linspace(-36.5, 36.5, 74)
        e_x = np.linspace(-0.5, 180.5, 182)
        return make_hist(abs(ad), ld, fit=fit, bins=[e_y, e_x])
    else:
        e_x = np.linspace(-180.5, 179.5, 361)
        e_y = np.linspace(-36.5, 36.5, 74)
        ad[ad > 179.5] -= 360
        return make_hist(ad, ld, fit=fit, bins=[e_y, e_x])

def make_hist(x_val, y_val, fit=spline_base.spline_pdf, 
            bins=None):
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
            Defaults to [np.linspace(-36.5,36.5,73),
                         np.linspace(-180.5,180.5,362)]
    """
    if not bins:
        bins =  (np.linspace(-36.5, 36.5, 73), np.linspace(-180.5, 180.5, 362))
    y_val = y_val[~np.isnan(y_val)]
    x_val = x_val[~np.isnan(x_val)]
    
    samples = zip(y_val, x_val)
    K = np.histogram2d(y_val, x_val, bins=bins)[0]
    K = K / sum(sum(K))
    
    if (fit is None):
        return K
    
    # Check if given attr is a function
    elif hasattr(fit, '__call__'):
        H = fit(np.array(samples), bins[0], bins[1], 
                nr_knots_y = 4, nr_knots_x = 10, hist=K)
        return H/H.sum()
        
    else:
        raise TypeError("Not a valid argument, insert spline function or None")


            
def anglendiff(fm, roll = 1, return_abs=False):
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
    
def first_saccade_dist(fm):
    """
        Computes the distribution of angle and length
        combinations that were made as first saccades
        
        Parameters:
            fm : ocupy.fixmat 
                The fixation data to be analysed
    
    """  
    ang, leng, ad, ld = anglendiff(fm, return_abs=True)
    screen_diag = int(ceil((fm.image_size[0]**2 + fm.image_size[1]**2)**0.5))/fm.pixels_per_degree
    y_arg = leng[0][np.roll(fm.fix == min(fm.fix), 1)]/fm.pixels_per_degree
    x_arg = reshift(ang[0][np.roll(fm.fix == min(fm.fix), 1)])
    bins = [range(screen_diag+1), np.linspace(-180.5, 180.5, 362)]
    return make_hist(x_arg, y_arg, fit=None, bins = bins)
    
def first_fixation_coordinates(fm):
    """
        Computes the distribution of coordinates that were chosen
        as first fixation locations
        
        Parameters:
            fm : ocupy.fixmat 
                The fixation data to be analysed
    
    """  
    ind = fm.fix == min(fm.fix)
    y_arg = fm.y[ind]
    x_arg = fm.x[ind]
    bins = [range(fm.image_size[0]+1), range(fm.image_size[1]+1)]
    return make_hist(x_arg, y_arg, fit=None, bins = bins)

    
def trajectory_lengths(fm):
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
    
def draw_from(cumsum, borders=None):
    """
    Draws a value from a cumulative sum.
    
    Parameters: 
        cumsum : array
            Cumulative sum from which shall be drawn.
        borders : array, optional
            If given, sets the value borders for entries in the cumsum-vector.
    Returns:
        int : Index of the cumulative sum element drawn.
    """
    if not borders is None:
        return (cumsum>=random.random()).nonzero()[0][0]
    else:
        return borders[(cumsum>=random.random()).nonzero()[0][0]]

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
    '''
    if type(I)==np.ndarray:    
        I[np.logical_or(I<-180.5, I>180.5)]  = (((I[np.logical_or(I<-180.5, I>180.5)])-180)%360)-180
    if type(I)==int or type(I)==float:
        if np.logical_or(I<-180.5, I>180.5):
            I = ((I-180)%360)-180
    '''
    
    return ((I-180)%360)-180

def calc_xy((x, y), angle, length):
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

if __name__ == '__main__':
    sim = FixGen('fixmat_photos.mat')
    #sim.set_path()
    simfm = sim.sample_many(num_samples=6263)

