#!/usr/bin/env python
"""This module implements the Datamat Structure for managing data structured in blocks (i.e. eye-tracking data.)"""
	import warnings
from os.path import join
from warnings import warn
from glob import glob

import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
from utils import snip_string_middle, isiterable
import h5py
from numpy import ma
class Datamat(object):
    """
    Represents grouped data.

    The datamat holds, filters and stores attributes that are grouped by some 
		event. For example, a group could be a trial in an experiment. The 
		attributes of this group might be associated with the subject's name or a
		trial condition.
     
    A datamat consists of lists called 'fields' which represent the raw
    underlying data and its attributes.

    A fixation, for example, has an x and a y position, but is also
    associated with the image that was being viewed while the fixation
    was made. The data can be accessed as attributes of the Datamat:

    datamat.x    # Returns a list of x-coordinates
    datamat.y    # Returns a list of y-coordinates

    In this case a single index into any one of these lists represents a 
    fixation:

    .. note:: It is never necessary to create a Datamat object directly. 
        This is handled by Datamat factories.
    """ 
    
    def __init__(self, categories = None, datamat = None, index = None):
        """
        Creates a new Datamat from an existing one

        Parameters:
            categories : optional, instance of stimuli.Categories,
                allows direct access to image data via the Datamat
            datamat : instance of datamat.Datamat, optional
                if given, the existing Datamat is copied and only those fixations
                that are marked True in index are retained.
            index : list of True or False or an iterable, same length as fields of Datamat
                Indicates which blocks should be used for the new Datamat and
                which should be ignored. If index is iterable it indexes all fields
								as if you would index a numpy array with index. The only exception is 
								that a datamat always holds arrays, never scalar values, as fields.

        TODO: thoroughly test that all indices work as expected (including slicing etc)
        """
        
        self._fields = []
        self._categories = categories.copy() if categories is not None else None
        self._parameters = {}
        self._num_fix = 0
        #warn('this needs to be thoroughly tested for indexes that are not boolean NumPy arrays!')
        if datamat is not None and index is not None:
            #index = index.reshape(-1,).astype(bool)
            #assert index.shape[0] == datamat._num_fix, ("Index vector for " +
            #    "filtering has to have the same length as the fields of the Datamat")
            #TODO: check this for slicing operations (fields will be views
            #rather than separate objects.
            if not isiterable(index):
                index = [index]
            self._fields = datamat._fields[:]
            for  field in self._fields:
                newfield = datamat.__dict__[field][index]
                num_fix = len(newfield)
                self.__dict__[field] = newfield
            self._parameters = datamat._parameters.copy()
            for (param, value) in datamat._parameters.iteritems():
                self.__dict__[param] = value
            self._num_fix = num_fix
    
    def __len__(self):
        return self._num_fix

    def __repr__(self):
        return 'Datamat(%i elements)' % (len(self))

    def __str__(self):
        desc = "Datamat with %i datapoints and the following data fields:\n" % (
                                                                    len(self))
        desc += "%s | %s | %s | %s \n" % ('Field Name'.rjust(20),
                                          'Length'.center(13), 
                                          'Type'.center(10), 
                                          'Values'.center(20))
        desc += "---------------------+---------------+------------+----------------\n"
        tmp_fieldnames = self._fields[:]
        tmp_fieldnames.sort()
        max_field_val_len = 40
        for field in tmp_fieldnames:
            if not self.__dict__[field].dtype == np.object:
                num_uniques = np.unique(self.__dict__[field])
                if len(num_uniques) > 5:
                    num_uniques = '%d unique'%(len(num_uniques))
                elif len(str(num_uniques)) > max_field_val_len:
                    per_val_len = (max_field_val_len // len(num_uniques))-1
                    if isinstance(num_uniques[0], str) or isinstance(num_uniques[0], unicode):
                        num_uniques = np.array([snip_string_middle(str(el),per_val_len, '..') for el in num_uniques])
            else:
                num_uniques = 'N/A'
            
            field_display_name = snip_string_middle(field, 20)
            desc += "%s | %s | %s | %s \n" % (field_display_name.rjust(20), 
                                    str(len(self.__dict__[field])).center(13),
                                    str(self.__dict__[field].dtype).center(10),
                                    str(num_uniques).center(20))
        desc += "---------------------+---------------+------------+----------------\n"
        desc += "%s | %s\n" % ('Parameter Name'.rjust(20), 'Value'.ljust(20))
        desc += "---------------------+---------------------------------------------\n"
        param_keys = self._parameters.keys()
        param_keys.sort()
        max_param_val_len = 13 + 3 + 10 + 3 + 20
        for param in param_keys:
            param_display_name = snip_string_middle(param, 20)
            desc += '%s | %s \n' % (param_display_name.rjust(20),
                                    snip_string_middle(str(self.__dict__[param]), max_param_val_len))
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
            
    def filter(self, index): #@ReservedAssignment
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
                Array-like that contains True for every element that
                passes the filter; else contains False
        Returns:
            datamat : Datamat Instance
        """
        return Datamat(categories=self._categories, datamat=self, index=index)

    def copy(self):
        """
        Returns a copy of the datamat.
        """
        return self.filter(np.ones(self._num_fix).astype(bool))


    def field(self, fieldname):
        """
        Return field fieldname. fm.field('x') is equivalent to fm.x.
        
        The '.' form (fm.x) is always easier in interactive use, but
        programmatically, this function can be useful if one has the field
        name in a variable.

        Parameters:
            fieldname : string
                The name of the field to be returned.
        """
        try:
            return self.__dict__[fieldname]
        except KeyError:
            raise ValueError('%s is not a field or parameter of the Datamat'
                    % fieldname)
            
    def save(self, path):
        """
        Saves Datamat to path.
        
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
            datamat : Datamat that is filtered according to one of the unique
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
    
    def add_field_like(self, name, like_array):
        """
        Add a new field to the Datamat with the dtype of the
        like_array and the shape of the like_array except for the first
        dimension which will be instead the field-length of this Datamat.
        """
        new_shape = list(like_array.shape)
        new_shape[0] = len(self)
        new_data = ma.empty(new_shape, like_array.dtype)
        new_data.mask = True
        self.add_field(name, new_data)

    def annotate (self, src_dm, data_field, key_field, take_first=True):
        """
        Adds a new field (data_field) to the Datamat with data from the
        corresponding field of another Datamat (src_dm).
        
				This is accomplished through the use of a key_field, which is
        used to determine how the data is copied.
        
				This operation corresponds loosely to an SQL join operation.

        The two Datamats are essentially aligned by the unique values
        of key_field so that each block element of the new field of the target
        Datamat will consist of those elements of src_dm's data_field
        where the corresponding element in key_field matches.
        
        If 'take_first' is not true, and there is not
        only a single corresponding element (typical usage case) then the
        target element value will be
        a sequence (array) of all the matching elements.
        
        The target Datamat (self) must not have a field name data_field
        already, and both Datamats must have key_field.
        
        The new field in the target Datamat will be a masked array to handle
        non-existent data.
       	
				TODO: Make example more generic, remove interoceptive reference
				TODO: Make standalone test
        Examples:
        
        >>> dm_intero = load_interoception_files ('test-ecg.csv', silent=True)
        >>> dm_emotiv = load_emotivestimuli_files ('test-bpm.csv', silent=True)
        >>> length(dm_intero)
        4
        >>> unique(dm_intero.subject_id)
        ['p05', 'p06']
        >>> length(dm_emotiv)
        3
        >>> unique(dm_emotiv.subject_id)
        ['p04', 'p05', 'p06']
        >>> 'interospective_awareness' in dm_intero.fieldnames()
        True
        >>> unique(dm_intero.interospective_awareness) == [0.5555, 0.6666]
        True
        >>> 'interospective_awareness' in dm_emotiv.fieldnames()
        False
        >>> dm_emotiv.copy_field(dm_intero, 'interospective_awareness', 'subject_id')
        >>> 'interospective_awareness' in dm_emotiv.fieldnames()
        True
        >>> unique(dm_emotiv.interospective_awareness) == [NaN, 0.5555, 0.6666]
        False
        """
        if key_field not in self._fields or key_field not in src_dm._fields:
            raise AttributeError('key field (%s) must exist in both Datamats'%(key_field))
        if data_field not in src_dm._fields:
            raise AttributeError('data field (%s) must exist in source Datamat' % (data_field))
        if data_field in self._fields:
            raise AttributeError('data field (%s) already exists in target Datamat' % (data_field))
        
        #Create a mapping of key_field value to data value.
        data_to_copy = dict([(x.field(key_field)[0], x.field(data_field)) for x in src_dm.by_field(key_field)])
        
        data_element = data_to_copy.values()[0]
        
        #Create the new data array of correct size.
        # We use a masked array because it is possible that for some elements
        # of the target Datamat, there exist simply no data in the source
        # Datamat. NaNs are fine as indication of this for floats, but if the
        # field happens to hold booleans or integers or something else, NaN
        # does not work.
        new_shape = [len(self)] + list(data_element.shape)
        new_data = ma.empty(new_shape, data_element.dtype)
        new_data.mask=True
        if np.issubdtype(new_data.dtype, np.float):
            new_data.fill(np.NaN) #For backwards compatibility, if mask not used
        
        #Now we copy the data. If the data to copy contains only a single value,
        # it is added to the target as a scalar (single value).
        # Otherwise, it is copied as is, i.e. as a sequence.
        for (key, val) in data_to_copy.iteritems():
            if take_first:
                new_data[self.field(key_field) == key] = val[0]
            else:
                new_data[self.field(key_field) == key] = val
        
        self.add_field(data_field, new_data)

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

    def add_parameter(self, name, value):
        """
        Adds a parameter to the existing Datamat.
        
        Fails if parameter with same name already exists or if name is otherwise
        in this objects ___dict__ dictionary.
        """
        if self._parameters.has_key(name):
            raise ValueError("'%s' is already a parameter" % (name))
        elif self.__dict__.has_key(name):
            raise ValueError("'%s' conflicts with the Datamat name-space" % (name))
        
        self.__dict__[name] = value
        self._parameters[name] = self.__dict__[name]

    def rm_parameter(self, name):
        """
        Removes a parameter to the existing Datamat.
        
        Fails if parameter doesn't exist.
        """
        if not self._parameters.has_key(name):
            raise ValueError("no '%s' parameter found" % (name))

        del self._parameters[name]
        del self.__dict__[name]
        
    def parameter_to_field(self, name):
        """
        Promotes a parameter to a field by creating a new array of same
        size as the other existing fields, filling it with the current
        value of the parameter, and then removing that parameter.
        """
        if not self._parameters.has_key(name):
            raise ValueError("no '%s' parameter found" % (name))
        if self._fields.count(name) > 0:
            raise ValueError("field with name '%s' already exists" % (name))
        
        data = np.array([self._parameters[name]]*self._num_fix)

        self.rm_parameter(name)
        self.add_field(name, data)
        
    def join(self, fm_new, minimal_subset=True):
        """
        Adds content of a new Datamat to this Datamat.
       
        If a parameter of the Datamats is not equal or does not exist
        in one, it is promoted to a field.
        
        If the two Datamats have different fields then the elements for the
        Datamats that did not have the field will be NaN, unless
        'minimal_subset' is true, in which case the mismatching fields will
        simply be deleted.
        
        Parameters
        fm_new : instance of Datamat
            This Datamat is added to the current one.
        minimal_subset : if true, remove fields which don't exist in both,
        	instead of using NaNs for missing elements (defaults to False)

        Capacity to use superset of fields added by rmuil 2012/01/30

        """
        # Check if parameters are equal. If not, promote them to fields.
        for (nm, val) in self._parameters.items():
            if fm_new._parameters.has_key(nm):
                if (val != fm_new._parameters[nm]):
                    self.parameter_to_field(nm)
                    fm_new.parameter_to_field(nm)
            else:
                self.parameter_to_field(nm)
        for (nm, val) in fm_new._parameters.items():
            if self._parameters.has_key(nm):
                if (val != self._parameters[nm]):
                    self.parameter_to_field(nm)
                    fm_new.parameter_to_field(nm)
            else:
                fm_new.parameter_to_field(nm)
        # Deal with mismatch in the fields
        # First those in self that do not exist in new...
        orig_fields = self._fields[:]
        for field in orig_fields:
            if not field in fm_new._fields:
                if minimal_subset:
                    self.rm_field(field)
                else:
										warnings.warn("This option is deprecated. Clean and Filter your data before it is joined.", DeprecationWarning)
                    fm_new.add_field_like(field, self.field(field))
        # ... then those in the new that do not exist in self.
        orig_fields = fm_new._fields[:]
        for field in orig_fields:
            if not field in self._fields:
                if minimal_subset:
                    fm_new.rm_field(field)
                else:
										warnings.warn("This option is deprecated. Clean and Filter your data before it is joined.", DeprecationWarning)
                    self.add_field_like(field, fm_new.field(field))

        # Concatenate fields
        for field in self._fields:
            self.__dict__[field] = ma.hstack((self.__dict__[field], 
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
        fields[field] = ma.array(value)
    for param, value in fm_group.attrs.iteritems():
        params[param] = value
    f.close()
    return VectorFactory(fields, params)

def VectorFactory(fields, parameters, categories = None):
    fm = Datamat(categories = categories)
    fm._fields = fields.keys()
    for (field, value) in fields.iteritems(): 
        fm.__dict__[field] = value 
    fm._parameters = parameters
    for (field, value) in parameters.iteritems(): 
       fm.__dict__[field] = value
    fm._num_fix = len(fm.__dict__[fields.keys()[0]])
    return fm

if __name__ == "__main__":

    import doctest
    doctest.testmod()
