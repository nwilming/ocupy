#!/usr/bin/env python
"""This module implements additional tools for eye-tracking data.

Most importantly, it houses some Factories to create fixation data
and the compute_fdm method.
"""
from os.path import join
from glob import glob
import h5py

import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

from datamat import DataMat, VectorDatamatFactory

class FixMat(DataMat):

    def add_feature_values(self, features):
        """
        Adds feature values of feature 'feature' to all fixations in 
        the calling fixmat.
        
        For fixations out of the image boundaries, NaNs are returned.
        The function generates a new attribute field named with the
        string in features that contains an np.array listing feature
        values for every fixation in the fixmat.
        
        .. note:: The calling fixmat must have been constructed with an 
        stimuli.Categories object
        
        Parameters:
            features : string
                list of feature names for which feature values are extracted.
        """
        if not 'x' in self.fieldnames():
            raise RuntimeError("""add_feature_values expects to find
        (x,y) locations in self.x and self.y. But self.x does not exist""")
 
        if not self._categories:
            raise RuntimeError(
            '''"%s" does not exist as a fieldname and the
            fixmat does not have a Categories object (no features 
            available. The fixmat has these fields: %s''' \
            %(features, str(self._fields))) 
        for feature in features:
            # initialize new field with NaNs
            feat_vals = np.zeros([len(self.x)]) * np.nan 
            for (cat_mat, imgs) in self.by_cat():
                for img in np.unique(cat_mat.filenumber).astype(int):
                    fmap = imgs[img][feature]
                    on_image = (self.x >= 0) & (self.x <= self.image_size[1])
                    on_image = on_image & (self.y >= 0) & (self.y <= self.image_size[0])
                    idx = (self.category == imgs.category) & \
                          (self.filenumber == img) & \
                          (on_image.astype('bool'))
                    feat_vals[idx] = fmap[self.y[idx].astype('int'), 
                        self.x[idx].astype('int')]
            # setattr(self, feature, feat_vals)
            self.add_field(feature, feat_vals)

    def make_reg_data(self, feature_list=None, all_controls=False):    
        """ 
        Generates two M x N matrices with M feature values at fixations for 
        N features. Controls are a random sample out of all non-fixated regions 
        of an image or fixations of the same subject group on a randomly chosen 
        image. Fixations are pooled over all subjects in the calling fixmat.
       
        Parameters : 
            all_controls : bool
                if True, all non-fixated points on a feature map are takes as
                control values. If False, controls are fixations from the same
                subjects but on one other randomly chosen image of the same
                category
            feature_list : list of strings
                contains names of all features that are used to generate
                the feature value matrix (--> number of dimensions in the 
                model). 
                ...note: this list has to be sorted !

        Returns : 
            N x M matrix of N control feature values per feature (M).
            Rows = Feature number /type
            Columns = Feature values
        """
        if not 'x' in self.fieldnames():
            raise RuntimeError("""make_reg_data expects to find
        (x,y) locations in self.x and self.y. But self.x does not exist""")

        on_image = (self.x >= 0) & (self.x <= self.image_size[1])
        on_image = on_image & (self.y >= 0) & (self.y <= self.image_size[0])
        assert on_image.all(), "All Fixations need to be on the image"
        assert len(np.unique(self.filenumber) > 1), "Fixmat has to have more than one filenumber"
        self.x = self.x.astype(int)
        self.y = self.y.astype(int)
        
        if feature_list == None:
            feature_list = np.sort(self._categories._features)
        all_act = np.zeros((len(feature_list), 1)) * np.nan
        all_ctrls = all_act.copy()
                            
        for (cfm, imgs) in self.by_cat():
            # make a list of all filenumbers in this category and then 
            # choose one random filenumber without replacement
            imfiles = np.array(imgs.images()) # array makes a copy of the list
            ctrl_imgs = imfiles.copy()
            np.random.shuffle(ctrl_imgs)
            while (imfiles == ctrl_imgs).any():
                np.random.shuffle(ctrl_imgs)
            for (imidx, img) in enumerate(imfiles):
                xact = cfm.x[cfm.filenumber == img]
                yact = cfm.y[cfm.filenumber == img]
                if all_controls:
                # take a sample the same length as the actuals out of every 
                # non-fixated point in the feature map
                    idx = np.ones(self.image_size)
                    idx[cfm.y[cfm.filenumber == img], 
                        cfm.x[cfm.filenumber == img]] = 0
                    yctrl, xctrl = idx.nonzero()
                    idx = np.random.randint(0, len(yctrl), len(xact))
                    yctrl = yctrl[idx]
                    xctrl = xctrl[idx]
                    del idx
                else:
                    xctrl = cfm.x[cfm.filenumber == ctrl_imgs[imidx]]
                    yctrl = cfm.y[cfm.filenumber == ctrl_imgs[imidx]]
                # initialize arrays for this filenumber
                actuals = np.zeros((1, len(xact))) * np.nan
                controls = np.zeros((1, len(xctrl))) * np.nan                
                
                for feature in feature_list:
                    # get the feature map
                    fmap = imgs[img][feature]
                    actuals = np.vstack((actuals, fmap[yact, xact]))
                    controls = np.vstack((controls, fmap[yctrl, xctrl]))
                all_act = np.hstack((all_act, actuals[1:, :]))
                all_ctrls = np.hstack((all_ctrls, controls[1:, :]))
        return (all_act[:, 1:], all_ctrls[:, 1:]) # first column was dummy 

def load(path):
    """
    Load fixmat at path.
    
    Parameters:
        path : string
            Absolute path of the file to load from.
    """
    f = h5py.File(path,'r')
    if 'Fixmat' in f:
      fm_group = f['Fixmat']
    else:
      fm_group = f['Datamat']
    fields = {}
    params = {}
    for field, value in fm_group.iteritems():
        fields[field] = np.array(value)
    for param, value in fm_group.attrs.iteritems():
        params[param] = value
    f.close()
    return VectorFixmatFactory(fields, params)


def compute_fdm(fixmat, fwhm=2, scale_factor=1):
    """
    Computes a fixation density map for the calling fixmat. 
    
    Creates a map the size of the image fixations were recorded on.  
    Every pixel contains the frequency of fixations
    for this image. The fixation map is smoothed by convolution with a
    Gaussian kernel to approximate the area with highest processing
    (usually 2 deg. visual angle).

    Note: The function does not check whether the fixmat contains
    fixations from different images as it might be desirable to compute
    fdms over fixations from more than one image.

    Parameters:
        fwhm :  float 
            the full width at half maximum of the Gaussian kernel used
            for convolution of the fixation frequency map.

        scale_factor : float
            scale factor for the resulting fdm. Default is 1. Scale_factor
            must be a float specifying the fraction of the current size.
        
    Returns:
        fdm  : numpy.array 
            a numpy.array of size fixmat.image_size containing
            the fixation probability for every location on the image.
    """
    # image category must exist (>-1) and image_size must be non-empty
    assert (len(fixmat.image_size) == 2 and (fixmat.image_size[0] > 0) and
        (fixmat.image_size[1] > 0)), 'The image_size is either 0, or not 2D'
    # check whether fixmat contains fixations
    if fixmat._num_fix == 0 or len(fixmat.x) == 0 or len(fixmat.y) == 0 :
        raise RuntimeError('There are no fixations in the fixmat.')
    assert not scale_factor <= 0, "scale_factor has to be > 0"
    # this specifies left edges of the histogram bins, i.e. fixations between
    # ]0 binedge[0]] are included. --> fixations are ceiled
    e_y = np.arange(0, np.round(scale_factor*fixmat.image_size[0]+1))
    e_x = np.arange(0, np.round(scale_factor*fixmat.image_size[1]+1))
    samples = np.array(zip((scale_factor*fixmat.y), (scale_factor*fixmat.x)))
    (hist, _) = np.histogramdd(samples, (e_y, e_x))
    kernel_sigma = fwhm * fixmat.pixels_per_degree * scale_factor
    kernel_sigma = kernel_sigma / (2 * (2 * np.log(2)) ** .5)
    fdm = gaussian_filter(hist, kernel_sigma, order=0, mode='constant')
    return fdm / fdm.sum()

def relative_bias(fm,  scale_factor = 1, estimator = None):
    """
    Computes the relative bias, i.e. the distribution of saccade angles 
    and amplitudes. 

    Parameters:
        fm : DataMat
            The fixation data to use
        scale_factor : double
    Returns:
        2D probability distribution of saccade angles and amplitudes.
    """
    assert 'fix' in fm.fieldnames(), "Can not work without fixation  numbers"
    excl = fm.fix - np.roll(fm.fix, 1) != 1

    # Now calculate the direction where the NEXT fixation goes to
    diff_x = (np.roll(fm.x, 1) - fm.x)[~excl]
    diff_y = (np.roll(fm.y, 1) - fm.y)[~excl]
       

    # Make a histogram of diff values
    # this specifies left edges of the histogram bins, i.e. fixations between
    # ]0 binedge[0]] are included. --> fixations are ceiled
    ylim =  np.round(scale_factor * fm.image_size[0])
    xlim =  np.round(scale_factor * fm.image_size[1])
    x_steps = np.ceil(2*xlim) +1
    if x_steps % 2 != 0: x_steps+=1
    y_steps = np.ceil(2*ylim)+1
    if y_steps % 2 != 0: y_steps+=1
    e_x = np.linspace(-xlim,xlim,x_steps)
    e_y = np.linspace(-ylim,ylim,y_steps)

    #e_y = np.arange(-ylim, ylim+1)
    #e_x = np.arange(-xlim, xlim+1)
    samples = np.array(zip((scale_factor * diff_y),
                             (scale_factor* diff_x)))
    if estimator == None:
        (hist, _) = np.histogramdd(samples, (e_y, e_x))
    else:
        hist = estimator(samples, e_y, e_x)
    return hist
     
                                             
def DirectoryFixmatFactory(directory, categories = None, glob_str = '*.mat'):
    """
    Concatenates all fixmats in dir and returns the resulting single
    fixmat.
    
    Parameters:
        directory : string
            Path from which the fixmats should be loaded
        categories : instance of stimuli.Categories, optional
            If given, the resulting fixmat provides direct access
            to the data in the categories object.
        glob_str : string
            A regular expression that defines which mat files are picked up
    Returns:
        f_all : instance of FixMat
            Contains all fixmats that were found in given directory
        
    """
    files = glob(join(directory,glob_str))
    if len(files) == 0:
        raise ValueError("Could not find any fixmats in " + 
            join(directory, glob_str))
    f_all = FixmatFactory(files.pop(), categories)
    for fname in files:
        f_current = FixmatFactory(fname, categories)
        f_all.join(f_current)
    return f_all


def FixmatFactory(fixmatfile, categories = None):
    """
    Loads a single fixmat (fixmatfile).
    
    Parameters:
        fixmatfile : string
            The matlab fixmat that should be loaded.
        categories : instance of stimuli.Categories, optional
            Links data in categories to data in fixmat.
    """
    data = loadmat(fixmatfile, struct_as_record = False)['fixmat'][0][0]
    num_fix = data.x.size
    
    # Get a list with fieldnames and a list with parameters
    fields = {}
    parameters = {}
    for field in data._fieldnames:
        if data.__getattribute__(field).size == num_fix:
            fields[field] = data.__getattribute__(field)
        else:            
            parameters[field] = data.__getattribute__(field)[0].tolist()
            if len(parameters[field]) == 1:
                parameters[field] = parameters[field][0]
    
    # Generate FixMat
    fixmat = FixMat(categories = categories)
    fixmat._fields = fields.keys()
    for (field, value) in fields.iteritems():
        fixmat.__dict__[field] = value.reshape(-1,) 

    fixmat._parameters = parameters
    fixmat._subjects = None
    for (field, value) in parameters.iteritems():
        fixmat.__dict__[field] = value
    fixmat._num_fix = num_fix
    return fixmat
    
def TestFixmatFactory(points = None, categories = [1], 
                filenumbers = [1], subjectindices = [1], params = None,
                categories_obj = None):
    """ 
    Returns a fixmat where the content is known. 

    Parameters:
        points : list, optional
            This list contains coordinates of fixations. I.e. list[0] contains x 
            and list[1] contains y. If omitted, the line that connects (0,0) 
            and (922,922) is used.
        category : list, default = [1]
            Category numbers to be used for the fixations. All fixations are
            repeated for every category.
        subjectindices : list, default = [1]
            Subjectindices to be used for the fixations. Every subjectindex will show
            up for every category in the test fixmat. 
        params : dictionary, optional
            A list of parameters that is set for the resulting fixmat. Defaults 
            are 'image_size':[922,1272] and 'pxels_per_degree':36

    """
    default_parameters = {'image_size':[922, 1272], 'pixels_per_degree':36}
    fixmat = FixMat(categories=categories_obj)
    fixmat.x = [] 
    fixmat.y = []
    fixmat.SUBJECTINDEX = []
    fixmat.filenumber = []
    fixmat.category = []
    fixmat.fix = [] 
    if not params is None:
        default_parameters.update(params)
    if points == None:
        points = [[ x for x in range(1, default_parameters['image_size'][0])],
                  [ x for x in range(1, default_parameters['image_size'][0])]]
    for cat in categories:
        for sub in subjectindices:
            for img in filenumbers:
                fixmat.x = np.hstack((fixmat.x, np.array(points[0])))
                fixmat.y = np.hstack((fixmat.y, np.array(points[1])))
                fixmat.SUBJECTINDEX = np.hstack((fixmat.SUBJECTINDEX, sub *
                    np.ones(len(points[0]))))
                fixmat.category = np.hstack((fixmat.category, cat *
                    np.ones(len(points[0]))))
                fixmat.filenumber = np.hstack((fixmat.filenumber, img *
                    np.ones(len(points[0]))))
                fixmat.fix = np.hstack((fixmat.fix, range(0,len(points[0]))))
 

    fixmat._fields = ['x', 'y', 'SUBJECTINDEX', 'filenumber', 'category', 'fix']
    fixmat._parameters = default_parameters
    for (field, value) in default_parameters.iteritems():
        fixmat.__dict__[field] = value
    fixmat._num_fix  = len(fixmat.x)
    return fixmat




def VectorFixmatFactory(fields, parameters, categories = None):
    fm = FixMat(categories = categories)
    fm._fields = fields.keys()
    for (field, value) in fields.iteritems(): 
        fm.__dict__[field] = value 
    fm._parameters = parameters
    for (field, value) in parameters.iteritems(): 
       fm.__dict__[field] = value
    fm._num_fix = len(fm.__dict__[fields.keys()[0]])
    return fm
