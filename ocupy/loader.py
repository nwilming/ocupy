#!/usr/bin/env python
"""This module abstracts access to stimuli from physical access to it."""

from os.path import join, isfile, isdir, split
from os import makedirs
from os import error as MKDirError

import Image

from scipy.io import loadmat, savemat
from scipy.misc import imread

import numpy as np

from utils import imresize


class Loader():
    """
    Represents an interface that encapsulates physical access to 
    images and features.
    """

    def __init__(self, impath = None, ftrpath = None):
        self.impath = impath
        self.ftrpath = ftrpath

    def path(self, *args):
        """
		Constructs path or URL to cat/img/ftr from *args
		"""
        raise NotImplementedError
     
    def get_image(self, cat, img):
        """
        Returns the image indexed by (cat,img) as numpy matrix.

        Input:
            cat: convertible to string with str().
                The category of the image that needs to be loaded.
            img: convertible to string with str()
                The image that needs to be loaded.
        Returns:
            numpy.ndarray: The image in matrix form. 
        """
        raise NotImplementedError
    
    def get_feature(self, cat, img, feature):
        """
        Returns the feature indexed by (cat,img,feature) as numpy matrix.

        Input:
            cat: Convertible to string with str()
                The category of the image for which the feature should be
                loaded.
            img: Convertible to string with str()
                The image to which the feature belongs.
            feature: string.
                Name of the feature.
        Returns:
            numpy.ndarray: The feature as a numpy matrix.
        """
        raise NotImplementedError

    
    def save_image(self, cat, img, data):
        """
        Save the image that is indexed by (cat, img) [optional].
        
        Input:
            cat: Convertible to string with str()
                The category of the image that needs to be saved.
            img: Convertible to string with str()
                The image identifier of the image to be saved.
            data: numpy.ndarray.
                Image in matrix form.

        """
        raise NotImplementedError("""This loader can not save images, 
            you have to use a different loader""")

    def save_feature(self, cat, img, feature, data):
       """
       Save the feature that is indexed by (cat, img, feature) [optional].
       
       Input:
           cat: Convertible to string with str()
               The category of the image of the feature that needs to be saved.
           img: Convertible to string with str()
               The image identifier of the feature to be saved.
           data: numpy.ndarray.
               Feature in matrix form.

       """
       raise NotImplementedError("""This loader can not save features, 
            you have to use a different loader""")
                
    def test_for_category(self, cat):
        """
        Tests if category cat exists.
        
        Input:
            cat: Convertible to string via str()
                The category to test for.
        Returns:
            boolean: True if the category exists and False otherwise.
        """
        raise NotImplementedError
    
    def test_for_image(self, cat, img):
        """
        Tests if image img in category cat exists
        
        Input:
            cat: Convertible to string via str()
                The category to test for.
            img: Convertible to string via str()
                The image to test for
        Returns:
            boolean: True if the image exists in the category and False otherwise.        
        """
        raise NotImplementedError
    
    def test_for_feature(self, cat, img, ftr):
        """
        Tests if feature ftr exists for image img in category cat.
        
        Input:
            cat: Convertible to string via str()
                The category to test for.
            img: Convertible to string via str()
                The image to test for
            feature: string
                Feature to test for.
        Returns:
            boolean: True if the feature exist and False otherwise.        
        """
        raise NotImplementedError

class LoadFromDisk(Loader):
    """A loader implementation that loads images and features 
    from the hard disk. Resizes images and features to a common 
    size if needed."""

    def __init__(self, impath=None, ftrpath=None, size=None):
        """
        Constructs a loader which loads images and features from disk and obeys
        the *catgegory/category_image.png* and 
        *category/feature/category_image.png* naming scheme.
        Parameters:
            impath : string
                base path where images are located
            ftrpath: string
                path where features are located. 
            size : (height, width)
                Size indicates how images and features should be resized. 
                Can be a tuple (height, width) which indicates the target
                size or a float. In the later case the target size is given by
                the original size * size.
        """
        if impath and not isdir(impath): 
            raise RuntimeError('Image path is not valid: %s'%impath)
        if ftrpath and not isdir(ftrpath): 
            raise RuntimeError('Feature path is not valid: %s'%ftrpath)
        Loader.__init__(self, impath, ftrpath)
        self.size = size

    def path(self, category = None, image = None, feature = None):
        """
		Constructs the path to categories, images and features.
    
        This path function assumes that the following storage scheme is used on
        the hard disk to access categories, images and features:
            - categories: /impath/category
            - images:     /impath/category/category_image.png
            - features:   /ftrpath/category/feature/category_image.mat
        
        The path function is called to query the location of categories, images
        and features before they are loaded. Thus, if your features are organized 
        in a different way, you can simply replace this method such that it returns
        appropriate paths' and the LoadFromDisk loader will use your naming
        scheme.
		"""
        filename = None
        if not category is None:
            filename = join(self.impath, str(category))
        if not image is None:
            assert not category is None, "The category has to be given if the image is given"
            filename = join(filename, 
                '%s_%s.png' % (str(category), str(image)))
        if not feature is None:
            assert category != None and image != None, "If a feature name is given the category and image also have to be given."
            filename = join(self.ftrpath, str(category), feature, 
                '%s_%s.mat' % (str(category), str(image)))
        return filename

    def get_image(self, cat, img):
        """ Loads an image from disk. """
        filename = self.path(cat, img)
        data = []
        if filename.endswith('mat'):
            data = loadmat(filename)['output']
        else:
            data = imread(filename)
        if self.size:
            return imresize(data, self.size)
        else:
            return data

    def get_feature(self, cat, img, feature):
        """
        Load a feature from disk.
        """
        filename = self.path(cat, img, feature)
        data = loadmat(filename)
        name = [k for k in data.keys() if not k.startswith('__')]
        if self.size:
            return imresize(data[name.pop()], self.size)
        return data[name.pop()]
        
    def test_for_category(self, cat):
        """Tests if category cat exists."""
        filename = self.path(cat)
        return isdir(filename)
    
    def test_for_image(self, cat, img):
        """Tests if image img in category cat exists"""
        filename = self.path(cat, img)
        return isfile(filename)
    
    def test_for_feature(self, cat, img, ftr):
        """Tests if feature ftr exists for image img in category cat """
        if not self.ftrpath:
            raise RuntimeError("Ftr. path was not set")
        filename = self.path(cat, img, ftr)
        return isfile(filename)
            

class SaveToDisk(LoadFromDisk):
    """
    A loader that adds functionality to save images or features.
    """       
    def save_image(self, cat, img, data):
        """Saves a new image."""
        filename = self.path(cat, img)
        mkdir(filename)
        if type(data) == np.ndarray:
            data = Image.fromarray(data).convert('RGB')
        data.save(filename)

    def save_feature(self, cat, img, feature, data):
        """Saves a new feature."""
        filename = self.path(cat, img, feature)
        mkdir(filename)
        savemat(filename, {'output':data})
      
def mkdir(filename):
    if not isdir(split(filename)[0]):
        try:
            makedirs(split(filename)[0])
        except MKDirError as error:
            if error.errno != 17: #errno==17 -> Dir already exists, which is OK
                raise

class TestLoader(Loader):
    """A loader used for testing the stimuli.* class. Simply 
    returns (cat, img) or (cat,img,feature)."""        
    
    def __init__(self, img_per_cat = None, features = None, size = (10, 10)):
        Loader.__init__(self)
        self.items = img_per_cat
        self.features = features
        self.size = size

    def get_image(self, cat, img):
        if self.items:
            if cat in self.items.keys() and img in self.items[cat]:
                return np.ones(self.size)
            else:
                raise IndexError('Category / Image not accessible')
        else:
            return (cat, img)
  
    def get_feature(self, cat, img, feature):
        # return a matrix (size is self.size) if feature was specified
        # beforehand. Also the image for the feature has to be specified
        # during __init__ of this loader
        if not feature in self.features:    
            raise RuntimeError('Feature not present')
        return np.ones(self.size)
        
    def test_for_category(self, cat):
        """Tests if category cat exists."""
        return not self.items == None and cat in self.items

    def test_for_image(self, cat, img):
        """Tests if image img in category cat exists"""
        return self.test_for_category(cat) and img in self.items[cat]

    def test_for_feature(self, cat, img, ftr):
        return self.test_for_image(cat, img) and ftr in self.features
