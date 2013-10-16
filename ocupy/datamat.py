#!/usr/bin/env python
"""
This module implements the Datamat Structure for managing data structured in 
blocks (i.e. eye-tracking data.)
"""
import numpy as np
from numpy import ma
import h5py
from warnings import warn
from utils import snip_string_middle, isiterable, all_same
import inspect
import numpy as np

try:
    dbg_lvl# @UndefinedVariable
    print 'dbg_lvl == %d'%dbg_lvl  # @UndefinedVariable
except NameError:
    dbg_lvl = 0

def dbg(lvl, msg):
    if lvl <= dbg_lvl:
        if lvl == 0:
            print '%s'%msg
        else:
            caller_name = inspect.stack()[1][3]
            print('dbg(%d|%s): %s'%(lvl, caller_name, msg))

def set_dbg_lvl(new_dbg_lvl):
    global dbg_lvl
    dbg_lvl = new_dbg_lvl

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
    fixation.

    .. note:: It is never necessary to create a Datamat object directly. 
        This is handled by Datamat factories.
    """ 
    
    def __init__(self, datamat = None, index = None):
        """
        Creates a new Datamat from an existing one

        Parameters:
            datamat : instance of datamat.Datamat, optional
                if given, the existing Datamat is copied and only those fixations
                that are marked True in index are retained.
            index : list of True or False or an iterable, same length as fields of Datamat
                Indicates which blocks should be used for the new Datamat and
                which should be ignored. If index is iterable it indexes all fields
                as if you would index a numpy array with index. The only exception is 
                that a datamat always holds arrays, never scalar values, as fields.

        TODO: thoroughly test that all indices work as expected (including slicing etc)
        
        NB: because of this usage of the constructor to filter also,
        and because of the non-intuitive self object in Python, and because of
        Python's multiple inheritance, sub-classing Datamat is a royal PITA.
        """
        
        self._fields = []
        self._parameters = {}
        self._num_fix = 0
        if datamat is not None and index is not None:
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
        desc = "Datamat with %i elements and the following data fields:\n" % (
                                                                    len(self))
        desc += "%s | %s | %s | %s\n" % ('Field Name'.rjust(20),
                                          'Length'.center(13), 
                                          'Type'.center(10), 
                                          'Values'.center(20))
        desc += "---------------------+---------------+------------+----------------\n"
        tmp_fieldnames = self._fields[:]
        tmp_fieldnames.sort()
        max_field_val_len = 40
        for field in tmp_fieldnames:
            value_str = '?'
            dat = self.__dict__[field]
            if len(self) >= 100000:
                value_str = 'N/A'
            elif not dat.dtype == np.object:
                unique_vals = np.unique(dat)
                if len(unique_vals) > 5:
                    value_str = '%d unique'%(len(unique_vals))
                elif len(str(unique_vals)) > max_field_val_len:
                    per_val_len = (max_field_val_len // len(unique_vals))-1
                    if isinstance(unique_vals[0], str) or isinstance(unique_vals[0], unicode):
                        value_str = str(np.array([snip_string_middle(str(el),per_val_len, '..') for el in unique_vals]))
                    else:
                        value_str = snip_string_middle(str(unique_vals), max_field_val_len, '..')
                else:
                    value_str = str(unique_vals)
            else: #object array
                if not isinstance(dat[0], np.ndarray):
                    value_str = type(dat[0]).__name__
                else:
                    value_str = str(dat[0].dtype)+' arrays'
            field_display_name = snip_string_middle(field, 20)
            desc += "%s | %s | %s | %s\n" % (field_display_name.rjust(20), 
                                    str(len(dat)).center(13),
                                    str(dat.dtype).center(10),
                                    str(value_str).center(20))
        desc += "---------------------+---------------+------------+----------------\n"
        desc += "%s | %s\n" % ('Parameter Name'.rjust(20), 'Value'.ljust(20))
        desc += "---------------------+---------------------------------------------\n"
        param_keys = self._parameters.keys()
        param_keys.sort()
        max_param_val_len = 13 + 3 + 10 + 3 + 20
        for param in param_keys:
            param_display_name = snip_string_middle(param, 20)
            desc += '%s | %s\n' % (param_display_name.rjust(20),
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
        
          >> print np.unique(fm.category)
          np.array([2,9])
            >> fm_filtered = fm[ fm.category == 9 ]
            >> print np.unique(fm_filtered)
            np.array([9])
    
        Parameters:
            index : array
                Array-like that contains True for every element that
                passes the filter; else contains False
        Returns:
            datamat : Datamat Instance
            
            NB: rmuil: should be using type(self) so that subclasses can use this function
            and don't get returned a bare Datamat. Tricky though.
        """
        return Datamat(datamat=self, index=index)

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
    
    def tohdf5(self, h5obj, name):
        fm_group = h5obj.create_group(name)
        for field in self.fieldnames():
            fm_group.create_dataset(field, data = self.__dict__[field])
        for param in self.parameters():
            fm_group.attrs[param]=self.__dict__[param]

                
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

    def set_param(self, key, value):
        """
        Set the value of a parameter.
        """
        self.__dict__[key] = value
        self._parameters[key] = value
                            
    def by_field(self, field, return_overall_idx=False):
        """
        Returns an iterator that iterates over unique values of field
        
        Parameters:
            field : string
                Filters the datamat for every unique value in field and yields 
                the filtered datamat.\
            return_overall_idx : boolean
                If true, will also return the index used to get each filtered
                DataMat, to determine where in the original DataMat the filtered
                DataMats come from. Use `ocupy.utils.expand_boolean_subindex()`
                to use this overall_idx.
        Returns:
            datamat : Datamat that is filtered according to one of the unique
                values in 'field'.
            overall_idx : boolean array - the index into the original DataMat
                used to retrieve this DataMat (only returned if
                 return_overall_idx==True)
        """
        for value in np.unique(self.__dict__[field]):
            overall_idx = self.__dict__[field] == value
            if return_overall_idx:
                yield (self.filter(overall_idx), overall_idx)
            else:
                yield self.filter(overall_idx)
    
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
        
        Examples:
            TODO: Make example more generic, remove interoceptive reference
            TODO: Make stand-alone test
        
        >> dm_intero = load_interoception_files ('test-ecg.csv', silent=True)
        >> dm_emotiv = load_emotivestimuli_files ('test-bpm.csv', silent=True)
        >> length(dm_intero)
        4
        >> unique(dm_intero.subject_id)
        ['p05', 'p06']
        >> length(dm_emotiv)
        3
        >> unique(dm_emotiv.subject_id)
        ['p04', 'p05', 'p06']
        >> 'interospective_awareness' in dm_intero.fieldnames()
        True
        >> unique(dm_intero.interospective_awareness) == [0.5555, 0.6666]
        True
        >> 'interospective_awareness' in dm_emotiv.fieldnames()
        False
        >> dm_emotiv.annotate(dm_intero, 'interospective_awareness', 'subject_id')
        >> 'interospective_awareness' in dm_emotiv.fieldnames()
        True
        >> unique(dm_emotiv.interospective_awareness) == [NaN, 0.5555, 0.6666]
        False
        """
        #warn(DeprecationWarning('Use datamat.merge() instead.'))
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
        aa = 1 if take_first else 0
        new_shape = [len(self)] + list(data_element.shape)[aa:]
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
        in this object's ___dict__ dictionary.
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
        
    def join(self, fm_new):
        """
        Adds content of a new Datamat to this Datamat, assuming same fields.
       
        If a parameter of the Datamats is not equal or does not exist
        in one, it is promoted to a field.
        
        If the two Datamats have different fields, the mismatching fields will
        simply be deleted.
        
        Parameters
        fm_new : instance of Datamat
            This Datamat is added to the current one.
        """
        # Check if parameters are equal. If not, promote them to fields.
        '''
        for (nm, val) in fm_new._parameters.items():
            if self._parameters.has_key(nm):
                if (val != self._parameters[nm]):
                    self.parameter_to_field(nm)
                    fm_new.parameter_to_field(nm)
            else:
                fm_new.parameter_to_field(nm)
        '''
        # Deal with mismatch in the fields
        # First those in self that do not exist in new...
        orig_fields = self._fields[:]
        for field in orig_fields:
            
            if not field in fm_new._fields:
                self.rm_field(field)
                warn("field '%s' doesn't exist in target DataMat, removing." % field)
        # ... then those in the new that do not exist in self.
        orig_fields = fm_new._fields[:]
        for field in orig_fields:
            if not field in self._fields:
                fm_new.rm_field(field)
                warn("field '%s' doesn't exist in source DataMat, removing." % field)
        # Concatenate fields
        for field in self._fields:
            self.__dict__[field] = ma.hstack((self.__dict__[field], 
                fm_new.__dict__[field]))

        # Update _num_fix
        self._num_fix += fm_new._num_fix
        
    def join_full(self, dm_new):
        """
        Combines the content of two Datamats.
       
        If a parameter of the Datamats is not equal or does not exist
        in one, it is promoted to a field.
        
        If the two Datamats have different fields then the elements for the
        Datamats that did not have the field will be NaN.
        
        Parameters
        dm_new : instance of Datamat
            This Datamat is added to the current one.

        Capacity to use superset of fields added by rmuil 2012/01/30

        """
        # Check if parameters are equal. If not, promote them to fields.
        for (nm, val) in self._parameters.items():
            if dm_new._parameters.has_key(nm):
                if (val != dm_new._parameters[nm]):
                    self.parameter_to_field(nm)
                    dm_new.parameter_to_field(nm)
            else:
                self.parameter_to_field(nm)
        for (nm, val) in dm_new._parameters.items():
            if self._parameters.has_key(nm):
                if (val != self._parameters[nm]):
                    self.parameter_to_field(nm)
                    dm_new.parameter_to_field(nm)
            else:
                dm_new.parameter_to_field(nm)
        # Deal with mismatch in the fields
        # First those in self that do not exist in new...
        orig_fields = self._fields[:]
        for field in orig_fields:
            if not field in dm_new._fields:
                dm_new.add_field_like(field, self.field(field))
        # ... then those in the new that do not exist in self.
        orig_fields = dm_new._fields[:]
        for field in orig_fields:
            if not field in self._fields:
                self.add_field_like(field, dm_new.field(field))

        # Concatenate fields
        for field in self._fields:
            self.__dict__[field] = ma.hstack((self.__dict__[field], 
                dm_new.__dict__[field]))

        # Update _num_fix
        self._num_fix += dm_new._num_fix 

#def merge (dm_l, dm_r, data_field, key_field):
#    """
#    Merges two DataMats.
#
#    This operation corresponds loosely to an SQL JOIN operation.
#
#    This is accomplished through the use of a key_field, which is
#    used to determine how the data is copied. The key_field must
#    provide a 1-to-1 mapping between the DataMats - that is, for
#    every unique value of src_dm.field(key_field) there must exist
#    only one element in self.field(key_field).
#
#    The two Datamats are essentially aligned by the unique values
#    of key_field so that each block element of the new field of the target
#    Datamat will consist of those elements of src_dm's data_field
#    where the corresponding element in key_field matches.
#
#    The target Datamat (self) must not have a field name data_field
#    already, and both Datamats must have key_field.
#
#    The new field in the target Datamat will be a masked array to handle
#    non-existent data.
#
#    Examples:
#        TODO: Make example more generic, remove interoceptive reference
#        TODO: Make stand-alone test
#
#    >> dm_intero = load_interoception_files ('test-ecg.csv', silent=True)
#    >> dm_emotiv = load_emotivestimuli_files ('test-bpm.csv', silent=True)
#    >> length(dm_intero)
#    4
#    >> unique(dm_intero.subject_id)
#    ['p05', 'p06']
#    >> length(dm_emotiv)
#    3
#    >> unique(dm_emotiv.subject_id)
#    ['p04', 'p05', 'p06']
#    >> 'interospective_awareness' in dm_intero.fieldnames()
#    True
#    >> unique(dm_intero.interospective_awareness) == [0.5555, 0.6666]
#    True
#    >> 'interospective_awareness' in dm_emotiv.fieldnames()
#    False
#    >> dm_emotiv.annotate(dm_intero, 'interospective_awareness', 'subject_id')
#    >> 'interospective_awareness' in dm_emotiv.fieldnames()
#    True
#    >> unique(dm_emotiv.interospective_awareness) == [NaN, 0.5555, 0.6666]
#    False
#    """
#    raise NotImplementedError()
#
#    if key_field not in dm_l._fields or key_field not in dm_r._fields:
#        raise AttributeError('key field (%s) must exist in both Datamats'%(
#                            key_field))
#    if data_field not in dm_r._fields:
#        raise AttributeError('data field (%s) must exist in left Datamat' % (
#                            data_field))
#    if data_field in dm_r._fields:
#        raise AttributeError('data field (%s) already exists in right Datamat' % (
#                            data_field))
#    if len(dm_l.field(key_field)) != len(np.unique(dm_l.field(key_field))):
#        raise AttributeError('non-unique elements exist in left DataMat for key field (%s)' % (
#                            key_field))
#    if len(dm_r.field(key_field)) != len(np.unique(dm_r.field(key_field))):
#        raise AttributeError('non-unique elements exist in right DataMat for key field (%s)' % (
#                            key_field))
#
#    #Create a mapping of key_field value to data value.
#    data_to_copy = dict([(x.field(key_field)[0], x.field(data_field)) for x in src_dm.by_field(key_field)])
#
#    data_element = data_to_copy.values()[0]
#
#    #Create the new data array of correct size.
#    # We use a masked array because it is possible that for some elements
#    # of the target Datamat, there exist simply no data in the source
#    # Datamat. NaNs are fine as indication of this for floats, but if the
#    # field happens to hold booleans or integers or something else, NaN
#    # does not work.
#    new_shape = [len(self)] + list(data_element.shape)
#    new_data = ma.empty(new_shape, data_element.dtype)
#    new_data.mask=True
#    if np.issubdtype(new_data.dtype, np.float):
#        new_data.fill(np.NaN) #For backwards compatibility, if mask not used
#
#    #Now we copy the data. If the data to copy contains only a single value,
#    # it is added to the target as a scalar (single value).
#    # Otherwise, it is copied as is, i.e. as a sequence.
#    for (key, val) in data_to_copy.iteritems():
#        new_data[self.field(key_field) == key] = val[0]
#
#    self.add_field(data_field, new_data)

def flatten(dm):
    """
    Takes a DataMat who's elements are arrays and flattens it so that
    the DataMat element is the lowest atom of data. Makes DataMat
    potentially extremely long, but eases merging, aligning, and maybe
    also analysis.
    """
    tmfields = dm.time_based_fields
    seqfields = []
    dbg(2, 'will flatten DataMat with %d elements.' % (len(dm)))
    for f in dm.fieldnames():
        if (dm.__dict__[f].dtype == np.object) and isiterable(dm.__dict__[f][0]):
            seqfields += [f]
            dbg(3, "seqfield: %s, %s, %s" % (f, 
                    type(dm.__dict__[f][0]),
                    dm.__dict__[f][0].dtype))
    
    nelements = 0
    for dmi in dm:
        elementn = [len(dmi.field(f)[0]) for f in seqfields]
        assert(all_same(elementn))
        nelements += elementn[0]
    dbg(2, 'flattened DataMat will contain %d elements' % (nelements))


def load(path):
    """
    Load datamat at path.
    
    Parameters:
        path : string
            Absolute path of the file to load from.
    """
    f = h5py.File(path,'r')
    dm = fromhdf5(f['Datamat'])
    f.close()
    return dm

def fromhdf5(fm_group):
    dm = {}
    params = {}
    for key, value in fm_group.iteritems():
        dm[key] = value
    for key, value in fm_group.attrs.iteritems():
        params[key] = value
    return VectorFactory(dm, params)
 
def VectorFactory(fields, parameters={}):
    """
    Creates a new DataMat based on 2 dictionaries: one for the fields and one
    for the parameters.
    
    Input:
        fields: Dictionary
            The values will be used as fields of the datamat and the keys
            as field names.
        parameters: Dictionary
            A dictionary whose values are added as parameters. Keys are used
            for parameter names.

    >>> new_dm = VectorFactory({'field1':ma.array([1,2,3,4]),\
    'field2':ma.array(['a','b','c','d'])},{'param1':'some parameter'})
    >>> new_dm
    Datamat(4 elements)
    >>> print new_dm # doctest:+ELLIPSIS
    Datamat with 4 elements and the following data fields:
              Field Name |     Length    |    Type    |        Values       
    ---------------------+---------------+------------+----------------
                  field1 |       4       |   int64    |      [1 2 3 4]      
                  field2 |       4       |    |S1     |  ['a' 'b' 'c' 'd']  
    ---------------------+---------------+------------+----------------
          Parameter Name | Value               
    ---------------------+---------------------------------------------
                  param1 | some parameter
    ...
    
    >>> new_dm = VectorFactory({'field1':ma.array([1,2,3,4])})
    >>> new_dm
    Datamat(4 elements)
    
    TODO: convert fields into arrays.
    """ 
    fm = Datamat()
    fm._fields = fields.keys()
    for (field, value) in fields.iteritems(): 
        fm.__dict__[field] = np.asarray(value)
    fm._parameters = parameters
    for (field, value) in parameters.iteritems(): 
        fm.__dict__[field] = value
    fm._num_fix = len(fm.__dict__[fields.keys()[0]])
    return fm

class AccumulatorFactory(object):
    
    def __init__(self):
        self.d = {}

    def update(self, a):
        if len(self.d.keys()) == 0:
            self.d = dict((k,[v]) for k,v in a.iteritems())
        else:
            # For all fields in a that are also in dict
            all_keys = set(a.keys() + self.d.keys())
            for key in all_keys:
                if key in a.keys():
                    value = a[key]
                if not key in self.d.keys():
                    # key is not yet in d. add it
                    self.d[key] = [np.nan]*len(self.d[self.d.keys()[0]])
                if not key in a.keys():
                    # key is not in new data. value should be nan
                    value = np.nan
                self.d[key].extend([value])                
    
    def get_dm(self, params = None):
        if params is None:
            params = {}
        return VectorFactory(self.d, params)

class DatamatAccumulator(object):
    def __init__(self):
        self.l = []

    def update(self, dm):
        self.l.append(dm.copy())

    def get_dm(self):
        # More efficient join
        length=0
        names = set(self.l[0].fieldnames())
        for d in self.l:
            length += len(d)
            names = names.intersection(d.fieldnames())
        dm_all = self.l[0].copy()
        dm_all._num_fix = length
        for f in names:
            dm_all.rm_field(f)
            dm_all.add_field(f, np.ones((length,)))
            offset = 0
            for d in self.l:
                val = d.field(f)
                dm_all.field(f)[offset:offset+len(d)] = val
                offset = offset+len(d)
        return dm_all
                

def DatamatFromRecordArray(arr):
    d = dict((k, arr[k][0][0].flatten()) for k in arr.dtype.fields)
    return VectorFactory(d,{})


if __name__ == "__main__":

    import doctest
    doctest.testmod()
