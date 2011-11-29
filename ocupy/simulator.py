#!/usr/bin/env python
"""This module implements a generator of data with given second-order dependencies"""

from math import radians, ceil, cos, sin 

import random

import ocupy
from ocupy import fixmat
import spline_base

from progressbar import ProgressBar, Percentage, Bar

import numpy as np



class AbstractSim(object):
    def __init__(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def parameters(self):
        raise NotImplementedError
    def finish(self):
        raise NotImplementedError

def makeHist(ad,ld,collapse=True, fit='spline'):
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
        xdim : list, optional
            Gives the range of values on the y-axis in case of the uncollapsed 
            histogram. Defaults to [-36,36].
    """
    ld = ld[~np.isnan(ld)]
    ad = reshift(ad[~np.isnan(ad)]) 
    samples = zip(ld,ad)

    if collapse: # von 0 bis 181
        e_y = np.linspace(-36.5,36.5,74)
        e_x = np.linspace(-0.5,180.5,182)
        K, xedges, yedges = np.histogram2d(ld[~np.isnan(ld)], ad[~np.isnan(ad)], bins=[e_y,e_x])
        K = K / sum(sum(K))
        K[:,0]*=2  
        K[:,-1]*=2
        # XXX fit parameter should not be a string, maybe None and spline_pdf? 
        # XXX Also this shoul not work when fit == 'foo'
        if (fit=='None'):
            return K
            
        elif (fit=='spline'):
            H = spline_base.spline_pdf(np.array(samples), e_y, e_x, nr_knots_y = 4, nr_knots_x = 10,hist=K)     
            return H/H.sum()
            
    else:
        e_x = np.linspace(-180.5,179.5,361)
        e_y = np.linspace(-36.5,36.5,74)
        ad[ad>179.5]-=360
        # XXX See above
        if (fit=='None'):
            K, xedges, yedges = np.histogram2d(ld[~np.isnan(ld)], ad[~np.isnan(ad)], bins=[e_y,e_x])
            K = K / sum(sum(K))
            #K[:,0]*=2  
            #K[:,-1]*=2
            return K
            
        elif (fit=='spline'):
            H = spline_base.spline_pdf(np.array(samples), e_y, e_x, nr_knots_y = 4, nr_knots_x = 10)        
            return H/H.sum()
    
class FixGen(AbstractSim):
    """
    Generates fixation data.
    The FixGen object creates a representation of the second order dependence structures
    between saccades contained in the fixmat given as input. It is then able to generate 
    and return a fixmat which replicates these dependencies, while consisting of different 
    fixations.
    In order to work, the initialized FixGen obejct has to initialize its data by calling
    the method initializeData():

            >>> gen = simulator.FixSim(fm)
            >>> gen.initializeData()
    
    Separating this time-consuming step from the initialization is helpful in cases of 
    parallelization.
    Data is generated upon calling the method sample_many(num_samples = 1000).  
    """
    # XXX default for fm_name should not be a path 
    def __init__(self, fm_name = '/home/student/s/sharst/Dropbox/NBP/fixmat_photos.mat'):
        """
        Creates a new FixGen object upon a certain fixmat
        
        Parameters: 
            fm_name: string or ocupy.fixmat
                The fixation data to replicate in fixmat format.
                Note: If the first fixation in the set was always kept centered,
                they have to be deleted prior to passing the fixmat to this function.               
        """
        # XXX Cleaner if it only accepts a fixmat
        if type(fm_name)==str:
            self.fm = fixmat.FixmatFactory(fm_name)
        elif type(fm_name)==ocupy.fixmat.FixMat:
            self.fm = fm_name
        else:
            raise ValueError("Not a valid argument, insert fixmat or path to fixmat")
        
        # XXX Specific hack for one fixmat, remove
        if (min(self.fm.fix)==1):
            self.firstfixcentered = False
        else:
            self.firstfixcentered = True
            
        
    def initializeData(self, fit='spline',full_H1=None):
        """
        Prepares the data to be replicated. Calculates the second-order length and angle
        dependencies between saccades and stores them in a fitted histogram.
        """
        ad, ld = anglendiff(self.fm, roll=1) 
        # XXX screen_diag never used
        screen_diag = int(ceil((np.sum(np.array(self.fm.image_size)**2)**.5)
                                                /self.fm.pixels_per_degree))
        # XXX I think not full_H1 is  None is nicer 
        if not full_H1 is None:
            self.full_H1 = makeHist(ad[0],ld[0]/self.fm.pixels_per_degree,
                                    collapse=False,fit=fit)
        else:
            self.full_H1 = full_H1
        # XXX  the arguments to compute_cumsum seem cryptic to me. 
        self.firstLengthsAngles_cumsum, self.firstLengthsAngles_shape = (
                                    compute_cumsum(self.fm, 'la'))
        self.probability_cumsum = np.cumsum(self.full_H1.flat)
        self.firstcoordinates_cumsum = compute_cumsum(self.fm,'coo')
        self.trajectoryLengths_cumsum, self.trajectoryLengths_borders = compute_cumsum(self.fm, 'len')
        
        # Counters for saccades that have to be canceled during the process
        self.canceled = 0
        self.minusSaccades = 0
        self.drawn = []
        
        
    def _calc_xy(self, (x,y), angle, length):
        """
        Calculates the coordinates after a specific saccade was made.
        
        Parameters:
            (x,y) : tuple of floats or ints
                The coordinates before the saccade was made
            angle : float or int
                The angle that the next saccade encloses with the display border
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
            
            Note: Either both prev_angle and prev_length have to be given or none;
            if only one parameter is given, it will be neglected.
        """
        # XXX to be honest, I often write if x == None  as well... :/
        if (prev_angle is None) or (prev_length is None):
            (length, angle) = np.unravel_index(drawFrom(self.firstLengthsAngles_cumsum),
                    self.firstLengthsAngles_shape)
            angle = angle-((self.firstLengthsAngles_shape[1]-1)/2)
        else:
            J, I = np.unravel_index(drawFrom(self.probability_cumsum), self.full_H1.shape)
            angle = reshift((I-self.full_H1.shape[1]/2) + prev_angle)
            self.drawn = np.append(self.drawn, J)
            length = prev_length + ((J-self.full_H1.shape[0]/2)*self.fm.pixels_per_degree)
        return angle, length
    
    def parameters(self):
        return {'fixmat':self.fm, 'sampling_dist':self.full_H1}

    # XXX Not needed?
    def finish(self):
        raise NotImplementedError
    
    def sample_many(self, num_samples = 2000):
        """
        Generates a given number of trajectories, using the method sample(). 
        Returns a fixmat with the generated data.
        
        Parameters:
            num_samples : int, optional
                The number of trajectories that shall be generated. If not given,
                it is set to 500.
        """     
        x = []
        y = []
        fix = []
        sample = []
        
        # XXX I'd get rid of this, at least the printing should be optional
        # XXX and turned off by default
        print "Simulating "+repr(num_samples)+" trajectories..."
        
        for s in xrange(0, num_samples):
            for i, (xs,ys) in enumerate(self.sample()):
                x.append(xs)
                y.append(ys)
                fix.append(i+1)
                sample.append(s)   
            
        fields = {'fix':np.array(fix),'y':np.array(y), 'x':np.array(x)}
        param = {'image_size':self.fm.image_size,'pixels_per_degree':self.fm.pixels_per_degree}
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
        sample_size = int(round(drawFrom(self.trajectoryLengths_cumsum, borders=self.trajectoryLengths_borders)))
        
        if (self.firstfixcentered == True):
            coordinates.append([self.fm.image_size[1]/2,self.fm.image_size[0]/2])
        else:
            K,L=(np.unravel_index(drawFrom(self.firstcoordinates_cumsum),[self.fm.image_size[0],self.fm.image_size[1]]))
            coordinates.append([L,K])
            
        fix.append(1)
        while len(coordinates) < sample_size:
            if len(lenghts) == 0 and len(angles) == 0:          
                angle, length = self._draw(self)                                                       
            else:
                angle, length = self._draw(prev_angle = angles[-1], prev_length = lenghts[-1])  
                        
            x, y = self._calc_xy(coordinates[-1], angle, length) 
            
            if (length<0):
                self.minusSaccades+=1
                pass # Drawn saccade length not possible
            else:
                coordinates.append([x,y])
                lenghts.append(length) 
                angles.append(angle)
                fix.append(fix[-1]+1)
        return coordinates
    
    
        
        
        
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
            The maximum order of the dependence structure that shall be analyzed -1.
            
                >>> anglendiff(fm, roll=2)   # Analyzes the data up to third order
            
            If none is given, only the second order properties are calculated.
        return_abs : boolean, optional
            By default, the method returns only length-angle difference pairs. 
            If return_abs is set to true, the length and angle absolutes are returned
            as well.
            
                >>> angles, lengths, angle_diffs, length_diffs = 
                            anglendiff(fm, return_abs = True)
    """
    
    angle_diffs = []
    length_diffs = []
    lengths = []
    angles  = []
    
    for r in range(1, roll+1):
        heights = (fm.y - np.roll(fm.y,r)).astype(float)
        widths = (fm.x - np.roll(fm.x,r)).astype(float)
        # XXX np.nan?
        heights[fm.fix<=min(fm.fix)+r-1]=float('nan')
        widths[fm.fix<=min(fm.fix)+r-1]=float('nan')
        
        lengths.append((widths**2+heights**2)**.5)
        angles.append(np.degrees(np.arctan2(heights,widths)))
        
        length_diffs.append(lengths[0] - np.roll(lengths[r-1],1))
        
        # -360: straight saccades, -180: return saccades, 0: straight saccades,
        # 180: return saccades, 360: no return saccades
        angle_diffs.append(angles[0] - np.roll(angles[r-1],1))
                
    if return_abs==True:
        return angles, lengths, angle_diffs, length_diffs
        
    else:
        return angle_diffs, length_diffs
            
def compute_cumsum(fm, arg):
    """
    Creates a probability distribution, transforms it to a single row array
    and calculates its cumulative sum.
    
    Parameters:
        fm : ocupy.fixmat
            The fixmat to take the data from.
        arg : string
            arg can take one of the following values:
            'la' : The probability distribution is calculated over the lengths and angles
                    of the very first saccades made on each image by the subjects.
            'coo' : The prob.dist. is calculated over the first coordinates 
                    fixated on each image.
            'len' : The prob.dist. is calculated over the amount of saccades per 
                    trajectory.
    Returns: 
        numpy.ndarray : Cumulative sum of the respective probability distribution 
    """
    # XXX I think the method should be refactored. 
    # XXX I'd probably write one method that takes input and ranges
    # XXX and then calls makeHist and returns the hist.

    # XXX The specifities in 'la', 'coo', 'len' can then go into three
    # XXX helper functions

    # XXX Also the name is a bit misleading, the method does much more
    # XXX than computing a cumsum, 99% actually goes into making the 
    # XXX probability distribution.

    if arg == 'la':
        ang, len, ad, ld = anglendiff(fm, return_abs=True)
        screen_diag = int(ceil((fm.image_size[0]**2+fm.image_size[1]**2)**0.5))
        y_arg = len[0][np.roll(fm.fix==min(fm.fix),1)]
        x_arg = reshift(ang[0][np.roll(fm.fix==min(fm.fix),1)])
        bins = [range(screen_diag+1), np.linspace(-180.5,180.5,362)]
    
    elif arg == 'coo':
        indexes = fm.fix==min(fm.fix)
        y_arg = fm.y[indexes]
        x_arg = fm.x[indexes]
        # XXX bins is ever used
        bins = [range(fm.image_size[0]+1), range(fm.image_size[1]+1)]
    
    elif arg == 'len':
        trajLen = np.roll(fm.fix,1)[fm.fix==min(fm.fix)]
        val, borders = np.histogram(trajLen, bins=1000)
        cumsum = np.cumsum(val.astype(float) / val.sum())
        return cumsum, borders
    
    else:
        raise ValueError("Not a valid argument, choose from 'la', 'coo' and 'len'.")
        
    H = makeHist(x_arg, y_arg, fit='None')
    return np.cumsum(H.flat), H.shape
    
def drawFrom(cumsum, borders=[]):
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
    if len(borders)==0:
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
    return ((I-180)%360)-180

    

    
    


if __name__ == '__main__':
    sim = FixSim('fixmat_photos.mat')
    #sim.set_path()
    simfm = sim.sample_many(num_samples=6263)

