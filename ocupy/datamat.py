#!/usr/bin/env python
"""This module implements the DataMat Structure for managing blocked data."""

from os.path import join
from warnings import warn
from glob import glob

import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

import h5py

def isiterable(some_object):
    try:
        iter(some_object)
    except TypeError:
        return False
    return True

class DataMat(object):
    """
    Represents blocked data.
    The DataMat object presents data, essentially, as discrete blocks. Each block
    is associated with attributes such as a subject's name or a trial condition.
    
    DataMat was FixMat, and so the typical 'block' of data is a fixation.
     
    A DataMat consists of lists called 'fields' which represent the raw
    underlying data and its attributes.
    
    A fixation, for example, has an x and a y position, but is also
    associated with the image that was being viewed while the fixation
    was made. The data can be accessed as attributes of the DataMat:

        >>> datamat.x    # Returns a list of x-coordinates
        >>> datamat.y    # Returns a list of y-coordinates

    In this case a single index into any one of these lists represents a 
    fixation:

        >>> (datamat.x[0], datamat.y[0])  

    .. note:: It is never necessary to create a DataMat object directly. 
        This is handled by DataMat factories.

    """ 
    
    def __init__(self, categories = None, datamat = None, index = None):
        """
        Creates a new DataMat from an existing one

        Parameters:
            categories : optional, instance of stimuli.Categories,
                allows direct access to image data via the DataMat
            datamat : instance of datamat.DataMat, optional
                if given, the existing DataMat is copied and only those fixations
                that are marked True in index are retained.
            index : list of True or False, same length as fields of DataMat
                Indicates which blocks should be used for the new DataMat and
                which should be ignored
        """
        self._subjects = []
        self._fields = []
        self._categories = categories
        self._parameters = {}
        if datamat is not None and index is not None:
            index = index.reshape(-1,).astype(bool)
            assert index.shape[0] == datamat._num_fix, ("Index vector for " +
                "filtering has to have the same length as the fields of the datamat")
            self._subjects = datamat._subjects
            self._fields = datamat._fields
            for  field in self._fields:
                self.__dict__[field] = datamat.__dict__[field][index]
            self._parameters = datamat._parameters
            for (param, value) in datamat._parameters.iteritems():
                self.__dict__[param] = value
                self._parameters[param] = self.__dict__[param]
            self._num_fix = index.sum()

    def __repr__(self):
        return 'DataMat(%s elements)' % (self._num_fix)

    def __str__(self):
        desc = "DataMat with %i blocks and the following data fields:\n" % (
                                                                    self._num_fix)
        desc += "%s | %s | %s | %s \n" % ('Field Name'.rjust(20),
                                          'Length'.center(13), 
                                          'Type'.center(10), 
                                          'Values'.center(20))
        desc += "---------------------+---------------+------------+----------------\n"
        for field in self._fields:
            if not self.__dict__[field].dtype == np.object:
                num_uniques = np.unique(self.__dict__[field])
                if len(num_uniques) > 5:
                    num_uniques = 'Many'
            else:
                num_uniques = 'N/A'
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
        Returns a filtered DataMat which only includes elements that are
        allowed by index 
        
		
        Parameters:
            index : numpy array
                A logical array that is True for elements that are in 
                the returned DataMat and False for elements that are 
                excluded.
        
        Notes:
            See datamat.filter for more info.
        """
        #if isinstance(key, IntType):
        #    self._categories[key]
        #else:
        return self.filter(key)
            
    def filter(self, index): #@ReservedAssignment
        """
        Filters a DataMat by different aspects.
        
        This function is a device to filter the DataMat by certain logical 
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
                Array-like that contains True for every element that
                passes the filter; else contains False
        Returns:
            datamat : DataMat Instance
        
        """
        return DataMat(categories=self._categories, datamat=self, index=index)

    def field(self, fieldname):
        """
        Return field fieldname. fm.field('x') is equivalent to fm.x.

        Parameters:
            fieldname : string
                The name of the field to be returned.
        """
        try:
            return self.__dict__[fieldname]
        except KeyError:
            raise ValueError('%s is not a field or parameter of the DataMat'
                    % fieldname)
            
    def save(self, path):
        """
        Saves DataMat to path.
        
        Parameters:
            path : string   
                Absolute path of the file to save to.
        """
        f = h5py.File(path, 'w')
        fm_group = f.create_group('DataMat')
        for field in self.fieldnames():
            fm_group.create_dataset(field, data = self.__dict__[field])
        for param in self.parameters():
            fm_group.attrs[param]=self.__dict__[param]
        f.close()

                
    def fieldnames(self):
        """
        Returns a list of data fields that are present in the DataMat.
        """
        return self._fields
            
    def parameters(self):
        """
        Return a list of parameters that are available. 
        
        .. note::Parameters refer to things like 'image_size', 'pixels_per_degree', 
            i.e. values that are valid for the entire DataMat.
        """
        return self._parameters
                            
    def by_field(self, field):
        """
        Returns an iterator that iterates over unique values of field
        
        Parameters:
            field : string
                Filters the datamat for every unique value in field and yields 
                the filtered datamat.
        Returns:
            datamat : DataMat that is filtered according to one of the unique
                values in 'field'.
        """
        for value in np.unique(self.__dict__[field]):
            yield self.filter(self.__dict__[field] == value)

    def by_cat(self): 
        """
        Iterates over categories and returns a filtered datamat. 
        
        If a categories object is attached, the images object for the given 
        category is returned as well (else None is returned).
        
        Returns:
            (datamat, categories) : A tuple that contains first the filtered
                datamat (has only one category) and second the associated 
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
        Iterates over categories and returns a filtered datamat. 
        
        If a categories object is attached, the images object for the given 
        category is returned as well (else None is returned).
        
        Returns:
            (datamat, categories) : A tuple that contains first the filtered
                datamat (has only one category) and second the associated 
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
        Add a new field to the datamat.

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
        Remove a field from the datamat.

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
        Adds content of a new datamat to this datamat.
        
        If the two datamats have different fields the minimal subset of both 
        are present after the join. Parameters of the datamats must be the 
        same.
 
        Parameters
        fm_new : Instance of DataMat
            This datamat is added to the current one. Can only contain data
            from one subject.

        """
        # Check if new datamat has only data from one subject
        if not len(np.unique(fm_new.SUBJECTINDEX))==1:
            raise (RuntimeError(
                """Can only join datamats if new datamat has data from only 
                one subject"""))
        
        # Check if parameters are equal
        #for ((n_cu, v_cu), (_, v_all)) in zip(fm_new._parameters.iteritems(),
        #                                        self._parameters.iteritems()):
        #    if not v_cu == v_all:
        #        raise (RuntimeError("""Parameter %s has value %s in current and
        #             value %s in new datamat""" %(n_cu, str(v_all), str(v_cu))))

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
        the calling DataMat.
        
        For fixations out of the image boundaries, NaNs are returned.
        The function generates a new attribute field named with the
        string in features that contains an np.array listing feature
        values for every fixation in the DataMat.
        
        .. note:: The calling DataMat must have been constructed with an 
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
            '''"%s" does not exist as a field and the
            DataMat does not have a Categories object (no features 
            available. The DataMat has these fields: %s''' \
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
        image. Fixations are pooled over all subjects in the calling DataMat.
       
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

        if (self.x < 2 * self.pixels_per_degree).any():
            warn('There are fixations within 2deg visual ' +
            'angle of the image border')
        on_image = (self.x >= 0) & (self.x <= self.image_size[1])
        on_image = on_image & (self.y >= 0) & (self.y <= self.image_size[0])
        assert on_image.all(), "All Fixations need to be on the image"
        assert len(np.unique(self.filenumber) > 1), "DataMat has to have more than one filenumber"
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
    Load datamat at path.
    
    Parameters:
        path : string
            Absolute path of the file to load from.
    """
    f = h5py.File(path,'r')
    fm_group = f['DataMat']
    fields = {}
    params = {}
    for field, value in fm_group.iteritems():
        fields[field] = np.array(value)
    for param, value in fm_group.attrs.iteritems():
        params[param] = value
    f.close()
    return VectorFactory(fields, params)


def compute_fdm(datamat, fwhm=2, scale_factor=1):
    """
    Computes a fixation density map for the calling DataMat. 
    
    Creates a map the size of the image fixations were recorded on.  
    Every pixel contains the frequency of fixations
    for this image. The fixation map is smoothed by convolution with a
    Gaussian kernel to approximate the area with highest processing
    (usually 2 deg. visual angle).

    Note: The function does not check whether the DataMat contains
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
            a numpy.array of size datamat.image_size containing
            the fixation probability for every location on the image.
    """
    # image category must exist (>-1) and image_size must be non-empty
    assert (len(datamat.image_size) == 2 and (datamat.image_size[0] > 0) and
        (datamat.image_size[1] > 0)), 'The image_size is either 0, or not 2D'
    assert datamat.pixels_per_degree, 'DataMat has to have a pixels_per_degree field'
    # check whether datamat contains fixations
    if datamat._num_fix == 0 or len( datamat.x) == 0 or len(datamat.y) == 0 :
        raise NoElements('There are no elements in the DataMat.')

    assert not scale_factor <= 0, "scale_factor has to be > 0"
    # this specifies left edges of the histogram bins, i.e. fixations between
    # ]0 binedge[0]] are included. --> fixations are ceiled
    e_y = np.arange(0, np.round(scale_factor*datamat.image_size[0]+1))
    e_x = np.arange(0, np.round(scale_factor*datamat.image_size[1]+1))
    samples = np.array(zip((scale_factor*datamat.y), (scale_factor*datamat.x)))
    (hist, _) = np.histogramdd(samples, (e_y, e_x))
    kernel_sigma = fwhm * datamat.pixels_per_degree * scale_factor
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
     
class NoElements(Exception):
    """
    Signals that a DataMat contains no elements 
    """
    def __init__(self, msg):
        self.msg = msg
    
    def __str__(self):
        return repr(self.msg)
        
                                             
def DirectoryFactory(directory, categories = None, glob_str = '*.mat'):
    """
    Concatenates all DataMats in the directory and returns the resulting single
    DataMat.
    
    Parameters:
        directory : string
            Path from which the DataMats should be loaded
        categories : instance of stimuli.Categories, optional
            If given, the resulting DataMat provides direct access
            to the data in the categories object.
        glob_str : string
            A regular expression that defines which mat files are picked up
    Returns:
        f_all : instance of DataMat
            Contains all DataMats that were found in given directory
        
    """
    files = glob(join(directory,glob_str))
    if len(files) == 0:
        raise ValueError("Could not find any DataMats in " + 
            join(directory, glob_str))
    f_all = MatFactory(join(files.pop()), categories)
    for fname in files:
        f_current = MatFactory(join(directory, fname), categories)
        f_all.join(f_current)

    return f_all


def MatFactory(datamatfile, categories = None):
    """
    Loads a single DataMat from a MatLab matfile.
    
    Parameters:
        datamatfile : string
            The matlab datamat that should be loaded.
        categories : instance of stimuli.Categories, optional
            Links data in categories to data in datamat.
    """
    data = loadmat(datamatfile, struct_as_record = False)['datamat'][0][0]
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
    
    # Generate DataMat
    datamat = DataMat(categories = categories)
    datamat._fields = fields.keys()
    for (field, value) in fields.iteritems():
        datamat.__dict__[field] = value.reshape(-1,) 

    datamat._parameters = parameters
    datamat._subjects = None
    for (field, value) in parameters.iteritems():
        datamat.__dict__[field] = value
    datamat._num_fix = num_fix
    return datamat
    
def TestFactory(points = None, categories = [1], 
                filenumbers = [1], subjectindices = [1], params = None,
                categories_obj = None):
    """ 
    Returns a datamat where the content is known. 

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
            up for every category in the test datamat. 
        params : dictionary, optional
            A list of parameters that is set for the resulting datamat. Defaults 
            are 'image_size':[922,1272] and 'pxels_per_degree':36

    """
    default_parameters = {'image_size':[922, 1272], 'pixels_per_degree':36}
    datamat = DataMat(categories=categories_obj)
    datamat.x = [] 
    datamat.y = []
    datamat.SUBJECTINDEX = []
    datamat.filenumber = []
    datamat.category = []
    datamat.fix = [] 
    if not params is None:
        default_parameters.update(params)
    if points == None:
        points = [[ x for x in range(1, default_parameters['image_size'][0])],
                  [ x for x in range(1, default_parameters['image_size'][0])]]
    for cat in categories:
        for sub in subjectindices:
            for img in filenumbers:
                datamat.x = np.hstack((datamat.x, np.array(points[0])))
                datamat.y = np.hstack((datamat.y, np.array(points[1])))
                datamat.SUBJECTINDEX = np.hstack((datamat.SUBJECTINDEX, sub *
                    np.ones(len(points[0]))))
                datamat.category = np.hstack((datamat.category, cat *
                    np.ones(len(points[0]))))
                datamat.filenumber = np.hstack((datamat.filenumber, img *
                    np.ones(len(points[0]))))
                datamat.fix = np.hstack((datamat.fix, range(0,len(points[0]))))
 

    datamat._fields = ['x', 'y', 'SUBJECTINDEX', 'filenumber', 'category', 'fix']
    datamat._parameters = default_parameters
    for (field, value) in default_parameters.iteritems():
        datamat.__dict__[field] = value
    datamat._num_fix  = len(datamat.x)
    return datamat

def VectorFactory(fields, parameters, categories = None):
    fm = DataMat(categories = categories)
    fm._fields = fields.keys()
    for (field, value) in fields.iteritems(): 
        fm.__dict__[field] = value 
    fm._parameters = parameters
    fm._subjects = None
    for (field, value) in parameters.iteritems():
        fm.__dict__[field] = value
    fm._num_fix = len(fm.__dict__[fields.keys()[0]])
    return fm
