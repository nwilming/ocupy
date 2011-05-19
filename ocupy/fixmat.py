#!/usr/bin/env python
"""This module implements the FixMat Structure for managing eye-tracking data."""

from os.path import join
from warnings import warn
import cPickle
from glob import glob

import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter


class FixMat(object):
    """
    Represents fixation data. 
    The fixmat object presents the fixation data as lists of values to the 
    user. In general fixation data consists of fixations that have several
    attributes. A fixation for example has a x and y position, but is also
    associated with the image that was being viewed while the fixation
    was made. The data can be accessed as attributes of the fixmat::

        >>> fixmat.x    # Returns a list of x-coordinates
        >>> fixmat.y    # Returns a list of y-coordinates

    In this case a single index into anyone of these lists represents a 
    fixation:

        >>> (fixmat.x[0], fixmat.y[0])  


    .. note:: It is never neccessary to create a FixMat object directly. 
        This is handled by Fixmat factories.

    """ 
    
    def __init__(self, categories = None, fixmat = None, index = None):
        """
        Creates a new fixmat from an existing one

        Parameters:
            categories : optional, instance of stimuli.Categories,
                allows direct access to image data via the fixmat
            fixmat : instance of fixmat.FixMat, optional
                if given, the existing fixmat is copied and only thos fixations
                that are marked True in index are retained.
            index : list of True or False, same length as fields of fixmat
                Indicates which fixations should be used for the new fixmat and
                which should be ignored
        """
        self._subjects = []
        self._fields = []
        self._categories = categories
        self._parameters = {}
        if fixmat is not None and index is not None:
            index = index.reshape(-1,).astype(bool)
            assert index.shape[0] == fixmat.x.shape[0], ("Index vector for " +
                "filtering has to have the same length as the fields of the fixmat")
            self._subjects = fixmat._subjects
            self._fields = fixmat._fields
            for  field in self._fields:
                self.__dict__[field] = fixmat.__dict__[field][index]
            self._parameters = fixmat._parameters
            for (param, value) in fixmat._parameters.iteritems():
                self.__dict__[param] = value
                self._parameters[param] = self.__dict__[param]
            self._num_fix = index.sum()
    
    def __str__(self):
        desc = "Fixmat with %i fixations and the following data fields:\n" % (
                                                                    len(self.x))
        desc += "%s | %s | %s | %s \n" % ('Field Name'.rjust(20),
                                          'Length'.center(13), 
                                          'Type'.center(10), 
                                          'Values'.center(20))
        desc += "---------------------+---------------+------------+----------------\n"
        for field in self._fields:
            num_uniques = np.unique(self.__dict__[field])
            if len(num_uniques) > 5:
                num_uniques = 'Many'
            desc += "%s | %s | %s | %s \n" % (field.rjust(20), 
                                    str(len(self.__dict__[field])).center(13),
                                    str(self.__dict__[field].dtype).center(10),
                                    str(num_uniques).center(20))
        desc += "---------------------+---------------+------------+----------------\n"
        desc += "%s | %s\n" % ('Parameter Name'.rjust(20), 'Value'.ljust(20))
        desc += "---------------------+---------------------------------------------\n"
        for param in self._parameters:
            desc += '%s | %s \n' % (param.rjust(20), str(self.__dict__[param]))
        return desc 
   
    def __getitem__(self, key):
        """
        Returns a filtered fixmat which only includes fixation that are
        allowed by index 
        
        Parameters:
            index : numpy array
                A logical array that is True for fixations that are in 
                the returned fixmat and False for fixations that are 
                excluded.
        
        Notes:
            See fixmat.filter for more info.
        """
        #if isinstance(key, IntType):
        #    self._categories[key]
        #else:
        return self.filter(key)
            
    def filter(self, index):
        """
        Filters a fixmat by different aspects.
        
        This function is a device to filter the fixmat by certain logical 
        conditions. It takes as input a logical array (contains only True
        or False for every fixation) and kicks out all fixations for which
        the array says False. The logical array can conveniently be created
        with numpy::
        
            >>> print np.unique(fm.category)
            np.array([2,9])
            >>> fm_filtered = fm[ fm.category == 9 ]
            >>> print np.unique(fm_filtered)
            np.array([9])
    
        Parameters:
            index : array
                Array-like that contains True for every fixtation that
                passes the filter; else contains False
        Returns:
            fixmat : FixMat Instance
        
        """
        return FixMat(categories=self._categories, fixmat=self, index=index)
            
            
    def save(self, path):
        """
        Saves fixmat to path.
        
        Parameters:
            path : string   
                Absolute path of the file to save to.
        """
        savefile = open(path,'wb')
        cPickle.dump( self, savefile)
   
    @staticmethod
    def load(path):
        """
        Load fixmat at path.
        
        Parameters:
            path : string
                Absolute path of the file to load from.
        """
        return cPickle.load(open(path, 'r'))
            
    def fieldnames(self):
        """
        Returns a list of data fields that are present in the fixmat.
        """
        return self._fields
            
    def parameters(self):
        """
        Return a list of parameters that are available. 
        
        .. note::Parameters refer to things like 'image_size', 'pixels_per_degree', 
            i.e. values that are valid for the entire fixmat.
        """
        return self._parameters
                            
    def by_field(self, field):
        """
        Returns an iterator that iterates over unique values of field
        
        Parameters:
            field : string
                Filters the fixmat for every unique value in field and yields 
                the filtered fixmat.
        Returns:
            fixmat : FixMat that is filtered according to one of the unique
                values in 'field'.
        """
        for value in np.unique(self.__dict__[field]):
            yield self.filter(self.__dict__[field] == value)

    def by_cat(self): 
        """
        Iterates over categories and returns a filtered fixmat. 
        
        If a categories object is attached, the images object for the given 
        category is returned as well (else None is returned).
        
        Returns:
            (fixmat, categories) : A tuple that contains first the filtered
                fixmat (has only one category) and second the associated 
                categories object (if it is available, None otherwise)
        """
        for value in np.unique(self.category):
            cat_fm = self.filter(self.category == value) 
            if self._categories:
                yield (cat_fm, self._categories[value]) 
            else: 
                yield (cat_fm, None) 
             
    def by_filenumber(self): 
        """
        Iterates over categories and returns a filtered fixmat. 
        
        If a categories object is attached, the images object for the given 
        category is returned as well (else None is returned).
        
        Returns:
            (fixmat, categories) : A tuple that contains first the filtered
                fixmat (has only one category) and second the associated 
                categories object (if it is available, None otherwise)
        """
        for value in np.unique(self.filenumber):
            file_fm = self.filter(self.filenumber == value) 
            if self._categories:
                yield (file_fm, self._categories[self.category[0]][value]) 
            else: 
                yield (file_fm, None)        
    
    def add_field(self, name, data):
        """
        Add a new field to the fixmat.

        Parameters:
            name : string
                Name of the new field
            data : list
                Data for the new field, must be same length as all other fields.
        """
        if name in self._fields:
            raise (ValueError(
                'Cannot add field %s. A field of the same name already exists.'
                %name))
        if not len(data) == self._num_fix:
            raise (ValueError(
                'Can not add field %s, data does not have correct length' 
                % (name)))
        self._fields.append(name)
        self.__dict__[name] = data
    
    def rm_field(self, name):
        """
        Remove a field from the fixmat.

        Parameters:
            name : string
                Name of the field to be removed
        """
        if not name in self._fields:
            raise (ValueError(
                'Cannot delete field %s. No such field exists'%name))
        self._fields.remove(name)
        del self.__dict__[name]

    def join(self, fm_new):
        """
        Adds content of a new fixmat to this fixmat.
        
        If the two fixmats have different fields the minimal subset of both 
        are present after the join. Parameters of the fixmats must be the 
        same.
 
        Parameters
        fm_new : Instance of Fixmat
            This fixmat is added to the current one. Can only contain data
            from one subject.

        """
        # Check if new fixmat has only data from one subject
        if not len(np.unique(fm_new.SUBJECTINDEX))==1:
            raise (RuntimeError(
                """Can only join fixmats if new fixmat has data from only 
                one subject"""))
        
        # Check if parameters are equal
        for ((n_cu, v_cu), (_, v_all)) in zip(fm_new._parameters.iteritems(),
                                                self._parameters.iteritems()):
            if not v_cu == v_all:
                raise (RuntimeError("""Parameter %s has value %s in current and
                     value %s in new fixmat""" %(n_cu, str(v_all), str(v_cu))))

        # Check if same fields are present, if not only use minimal subset
        new_fields = []
        for field in self._fields:
            if not field in fm_new._fields:
                # field does not exist in f_current, del it from f_all
                delattr(self, field)
            else:
                # field exists, so keep it. afraid of just deleting it while
                # iterating over the list so reconstruct a new one
                new_fields.append(field)
        self._fields = new_fields
        # Subjectindices must be unique, if indices in f_current are contained
        # in f_all set them to an arbitrary number
        if fm_new.SUBJECTINDEX[0] in self.SUBJECTINDEX:
            fm_new.SUBJECTINDEX = np.ones(fm_new.SUBJECTINDEX.shape) * \
            (max(self.SUBJECTINDEX)+1)
        
        # Concatenate fields
        for field in self._fields:
            self.__dict__[field] = np.hstack((self.__dict__[field], 
                fm_new.__dict__[field]))

        # Update _num_fix
        self._num_fix += fm_new._num_fix 

        
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
        if (self.x < 2 * self.pixels_per_degree).any():
            warn('There are fixations within 2deg visual ' +
            'angle of the image border')
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
    assert fixmat.pixels_per_degree, 'Fixmat has to have a pixels_per_degree field'
    # check whether fixmat contains fixations
    if len(fixmat.x) == 0:
        raise NoFixations('There are no fixations in the fixmat.')

    assert not scale_factor <= 0, "scale_factor has to > 0"
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


def relative_bias(fm, center=None, radius=None, scale_factor = 1):
    """
    Computes the relative bias, i.e. the distribution of saccade angles 
    and amplitudes. 

    Parameters:
        fm : FixMat
            The fixation data to use
        center : 2D Point (Tuple)
            Only saccades that are at most 'radius' px. away from 'center'
            are considered for computing the distribution.
        radius : double
            Only saccades that are at most 'radius' px. away from 'center'
            are considered for computing the distribution.
        scale_factor : double
    Returns:
        2D probability distribution of saccade angles and amplitudes.
    """
    assert 'fix' in fm.fieldnames(), "Can not work without fixation  numbers"
    if not center:
        x = fm.image_size[1]/2.0
        y = fm.image_size[0]/2.0
    else:
        (y, x) = center
    if not radius:
        radius = (x**2 + y**2) ** .5
   
    # Add two fields to our fixmat: distance and difference
    # calculate how far this fixation is from the center of interest,
    dist = ((fm.x-x)**2 + (fm.y-y)**2)**.5
    # Now calculate the direction where the NEXT fixation goes to
    diff_x = np.roll(fm.x, 1) - fm.x
    diff_y = np.roll(fm.y, 1) - fm.y
    # We can not include those fixation pairs that are 
    # a) not on the same stimulus and
    # b) where fixations are not consecutive
    excl = fm.fix - np.roll(fm.fix, 1) != 1
    # Add to fixmat
    fm.add_field('dist', dist)
    fm.add_field('diff_x', diff_x)
    fm.add_field('diff_y', diff_y)
    
    # Find all fixations that are within radius around (y,x)
    f_filt = fm[ (~excl) & (fm.dist <= radius)]

    # Make a histogram of diff values
    # this specifies left edges of the histogram bins, i.e. fixations between
    # ]0 binedge[0]] are included. --> fixations are ceiled
    ylim =  np.round(scale_factor * fm.image_size[0])
    xlim =  np.round(scale_factor * fm.image_size[1])
    e_y = np.arange(-ylim, ylim+1)
    e_x = np.arange(-xlim, xlim+1)
    samples = np.array(zip((scale_factor * f_filt.diff_y),
                             (scale_factor*f_filt.diff_x)))
    (hist, _) = np.histogramdd(samples, (e_y, e_x))
    return hist
     
class NoFixations(Exception):
    """
    Signals that a FixMat contains no Fixations 
    """
    def __init__(self, msg):
        self.msg = msg
    
    def __str__(self):
        return repr(self.msg)
        
                                             
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
    f_all = FixmatFactory(join(files.pop()), categories)
    for fname in files:
        f_current = FixmatFactory(join(directory, fname), categories)
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

