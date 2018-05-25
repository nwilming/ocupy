'''
Created on Feb 1, 2012

@author: rmuil

Things here should eventually be incorporated into the DataMat class itself. 
'''
from .utils import factorise_strings
from numpy import array

def factorise_field(dm, field_name, boundary_char = None, parameter_name=None):
    """This removes a common beginning from the data of the fields, placing
    the common element in a parameter and the different endings in the fields.
    
    if parameter_name is None, then it will be <field_name>_common.
    
    So far, it's probably only useful for the file_name.
    
    TODO: remove field entirely if no unique elements exist.
    """
    
    old_data = dm.field(field_name)
    
    if isinstance(old_data[0], str) or isinstance(old_data[0], str):
        (new_data, common) = factorise_strings(old_data, boundary_char)
        new_data = array(new_data)
    else:
        raise NotImplementedError('factorising of fields not implemented for anything but string/unicode objects')
    
    if len(common) > 0:
        dm.__dict__[field_name] = new_data
        if parameter_name is None:
            parameter_name = field_name + '_common'
        dm.add_parameter(parameter_name, common)