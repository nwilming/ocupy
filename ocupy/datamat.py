#!/usr/bin/env python
"""This module implements the DataMat Structure for managing 
structured data (i.e. eye-tracking data.)"""
import numpy as np
import h5py

class DataMat(object):
    """
    Represents structured data (i.e. fixations). 
    The datamat object presents data as lists of values to the 
    user. In general datamats consists of datapoints that have several
    attributes. A fixation for example has a x and y position, but is also
    associated with the image that was being viewed while the fixation
    was made. The data can be accessed as attributes of the datamat::

        >>> datamat.x    # Returns a list of x-coordinates
        >>> datamat.y    # Returns a list of y-coordinates

    In this case a single index into anyone of these lists represents a 
    fixation:

        >>> (datamat.x[0], datamat.y[0])  


    .. note:: It is never neccessary to create a FixMat object directly. 
        This is handled by Datamat factories.

    """ 
    
    def __init__(self, categories = None, datamat = None, index = None):
        """
        Creates a new datamat from an existing one

        Parameters:
            categories : optional, instance of stimuli.Categories,
                allows direct access to image data via the datamat
            datamat : instance of datamat.FixMat, optional
                if given, the existing datamat is copied and only thos datapoints
                that are marked True in index are retained.
            index : list of True or False, same length as fields of datamat
                Indicates which datapoints should be used for the new datamat and
                which should be ignored
        """
        self._fields = []
        self._categories = categories
        if datamat is not None and index is not None:
            index = index.reshape(-1,).astype(bool)
            assert index.shape[0] == datamat._num_fix, ("Index vector for " +
                "filtering has to have the same length as the fields " + 
                "of the datamat")
            self._fields = [field for field in datamat._fields]
            for  field in self._fields:
                self.__dict__[field] = datamat.__dict__[field][index]
            self._parameters = {}
            for (param, value) in datamat._parameters.iteritems():
                self.__dict__[param] = value
                self._parameters[param] = self.__dict__[param]
            self._num_fix = index.sum()
    
    def __str__(self):
        desc = "Datamat with %i datapoints and the following data fields:\n" % (
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
        Returns a filtered datamat which only includes datapoints that are
        allowed by index 
        
		
        Parameters:
            index : numpy array
                A logical array that is True for datapoints that are in 
                the returned datamat and False for datapoints that are 
                excluded.
        
        Notes:
            See datamat.filter for more info.
        """
        return self.filter(key)
            
    def filter(self, index):
        """
        Filters a datamat by different aspects.
        
        This function is a device to filter the datamat by certain logical 
        conditions. It takes as input a logical array (contains only True
        or False for every datapoint) and kicks out all datapoints for which
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
            datamat : FixMat Instance
        
        """
        return DataMat(categories=self._categories, datamat=self, index=index)

    def copy(self):
        """
        Returns a copy of the datamat.
        """
        return self.filter(np.ones(self._num_fix).astype(bool))


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
            raise ValueError('%s is not a field or parameter of the datamat'
                    % fieldname)
            
    def save(self, path):
        """
        Saves datamat to path.
        
        Parameters:
            path : string   
                Absolute path of the file to save to.
        """
        f = h5py.File(path, 'w')
        fm_group = f.create_group('Datamat')
        for field in self.fieldnames():
            fm_group.create_dataset(field, data = self.__dict__[field])
        for param in self.parameters():
            fm_group.attrs[param]=self.__dict__[param]
        f.close()

                
    def fieldnames(self):
        """
        Returns a list of data fields that are present in the datamat.
        """
        return self._fields
            
    def parameters(self):
        """
        Return a list of parameters that are available. 
        
        .. note::Parameters refer to things like 'image_size', 'pixels_per_degree', 
            i.e. values that are valid for the entire datamat.
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
            datamat : FixMat that is filtered according to one of the unique
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
        fm_new : Instance of Datamat
            This datamat is added to the current one. Can only contain data
            from one subject.

        """
        
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
        
        # Concatenate fields
        for field in self._fields:
            self.__dict__[field] = np.hstack((self.__dict__[field], 
                fm_new.__dict__[field]))

        # Update _num_fix
        self._num_fix += fm_new._num_fix 


def load(path):
    """
    Load datamat at path.
    
    Parameters:
        path : string
            Absolute path of the file to load from.
    """
    f = h5py.File(path,'r')
    fm_group = f['Datamat']
    fields = {}
    params = {}
    for field, value in fm_group.iteritems():
        fields[field] = np.array(value)
    for param, value in fm_group.attrs.iteritems():
        params[param] = value
    f.close()
    return VectorDatamatFactory(fields, params)


def VectorDatamatFactory(fields, parameters, categories = None):
    fm = DataMat(categories = categories)
    fm._fields = fields.keys()
    for (field, value) in fields.iteritems(): 
        fm.__dict__[field] = value 
    fm._parameters = parameters
    for (field, value) in parameters.iteritems(): 
       fm.__dict__[field] = value
    fm._num_fix = len(fm.__dict__[fields.keys()[0]])
    return fm
