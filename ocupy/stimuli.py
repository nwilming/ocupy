#!/usr/bin/env python
"""This module implements different model evaluation measures."""

import os

import numpy as np

from ocupy.utils import ismember


class Categories(object):
    """This class represents different categories of stimuli."""
    def __init__(self, loader, img_per_cat, features = None, fixations = None):
        self.loader = loader
        self._features = features
        self._fixations = fixations
        self._categories = {}
        self._img_per_cat = img_per_cat
        for (k, imgs) in img_per_cat.iteritems():
            self._categories.update({k:Images(loader, 
                imgs, k,features,fixations)})
    
    def content(self):
        """
        Return category numbers and associated file numbers.
        """
        return self._img_per_cat

    def __contains__(self, key):
        return key in self._categories.keys()
     
    def __iter__(self):
        for i in self._categories.values():
            yield i
    
    def categories(self):
        """
        Returns a list of category numbers
        """
        return self._categories.keys()

    def __getitem__(self, key):
        if not key in self._categories.keys():
            raise IndexError('The requested Category was not '
                + 'specified beforehand')
        return self._categories[key]
        
    @property
    def fixations(self):
        ''' Filter the fixmat such that it only contains fixations on images
        in categories that are also in the categories object'''
        if not self._fixations:
            raise RuntimeError('This Images object does not have'
                +' an associated fixmat')
        if len(self._categories.keys()) == 0:
            return None
        else:
            idx = np.zeros(self._fixations.x.shape, dtype='bool')
            for (cat, _) in self._categories.iteritems():
                idx = idx | ((self._fixations.category == cat))
            return self._fixations[idx]

class Images(object):
    """
    Represents all stimuli that are in a category.
    """ 
    def __init__(self, loader, images, category, 
                 features = None, fixations = None):
        self.loader = loader
        self.category = category
        self._features = features
        self._fixations = fixations
        self._images = {}
        for img in images:
            self._images.update({img: Image(loader, category, 
                img, features, fixations)})
 
    def __contains__(self, key):
        return key in self._images.keys()
     
    def __iter__(self):
        for i in self._images.values():
            yield i

    def images(self):
        """
        Returns a list image numbers.
        """
        return self._images.keys()
       
    def __getitem__(self, key):
        if not key in self._images.keys():
            raise IndexError('The requested Image was not specified')
        return self._images[key]
    
    @property
    def fixations(self):
        if not self._fixations:
            raise RuntimeError('This Images object does not have'
                +' an associated fixmat')
        return self._fixations[(self._fixations.category == self.category) &
                ismember(self._fixations.filenumber, self._images.keys())]   

class Image(object):
    """
    Represents a single stimulus
    """
    def __init__(self, loader, category, image, 
                 features = None, fixations = None):
        self.loader = loader
        self.category = category
        self.image = image
        if not features:
            self._features = []
        else:
            self._features = features
        self._fixations = fixations
     
    def __contains__(self, key):
        return key in self._features
     
    def __iter__(self):
        for i in self._features:
            yield self[i]

    def features(self):
        """
        Returns features associated to this image.
        """
        return self._features
   
    @property
    def data(self):
        """
        Returns the image data for this ctegory/image combination.
        """
        return self.loader.get_image(self.category, self.image)
    
    @data.setter
    def data(self, value):
        """
        Saves a new image to disk
        """
        self.loader.save_image(self.category, self.image, value)
    
    def __getitem__(self, key):
        if not key in self._features:
            raise IndexError('The feature was not specified beforehand')
        if not self.loader.test_for_feature(self.category, self.image, key):
            raise IOError('Cannot load Feature %s' % (key, ))
        feature = self.loader.get_feature(self.category, self.image, key)
        return feature
    
    def __setitem__(self, key, value):
        # when models make new predictions it can be usefull to add new items
        # to the feature set. I.e. a prediciton of the model.
        if not key in self._features:
            self._features.append(key)
 
        self.loader.save_feature(int(self.category), 
                                 int(self.image), 
                                 key, 
                                 value)

    @property
    def fixations(self):
        """
        Returns all fixations that are on this image.
        A precondition for this to work is that a fixmat 
        is associated with this Image object.
        """
        if not self._fixations:
            raise RuntimeError('This Images object does not have'
                +' an associated fixmat')
        return self._fixations[(self._fixations.category == self.category) &
                               (self._fixations.filenumber == self.image)]


def FixmatStimuliFactory(fm, loader):
    """
    Constructs an categories object for all image / category 
    combinations in the fixmat.
    
    Parameters:
        fm: FixMat
            Used for extracting valid category/image combination.
        loader: loader
            Loader that accesses the stimuli for this fixmat
 
    Returns:
        Categories object
    """
    # Find all feature names
    features = [] 
    if loader.ftrpath:
        assert os.access(loader.ftrpath, os.R_OK)   
        features = os.listdir(os.path.join(loader.ftrpath, str(fm.category[0])))
    # Find all images in all categories   
    img_per_cat = {}
    for cat in np.unique(fm.category):
        if not loader.test_for_category(cat):
            raise ValueError('Category %s is specified in fixmat but '%(
                                str(cat) + 'can not be located by loader'))
        img_per_cat[cat] = []
        for img in np.unique(fm[(fm.category == cat)].filenumber):
            if not loader.test_for_image(cat, img):
                raise ValueError('Image %s in category %s is '%(str(cat), 
                    str(img)) + 
                    'specified in fixmat but can be located by loader')
            img_per_cat[cat].append(img)
            if loader.ftrpath:
                for feature in features:
                    if not loader.test_for_feature(cat, img, feature):
                        raise RuntimeError(
                            'Feature %s for image %s' %(str(feature),str(img)) +
                            ' in category %s ' %str(cat) +
                            'can not be located by loader') 
    return Categories(loader, img_per_cat = img_per_cat,
         features = features, fixations = fm)

def DirectoryStimuliFactory(loader):
    """
    Takes an input path to the images folder of an experiment and generates
    automatically the category - filenumber list needed to construct an
    appropriate _categories object.
    
    Parameters :
        loader : Loader object which contains 
            impath : string
                path to the input, i.e. image-, files of the experiment. All
                subfolders in that path will be treated as categories. If no
                subfolders are present, category 1 will be assigned and all 
                files in the folder are considered input images. 
                Images have to end in '.png'.
            ftrpath : string
                path to the feature folder. It is expected that the folder
                structure corresponds to the structure in impath, i.e. 
                    ftrpath/category/featurefolder/featuremap.mat
                Furthermore, features are assumed to be the same for all 
                categories.
    """
    impath = loader.impath
    ftrpath = loader.ftrpath
    # checks whether user has reading permission for the path
    assert os.access(impath, os.R_OK)
    assert os.access(ftrpath, os.R_OK)    

    # EXTRACTING IMAGE NAMES
    img_per_cat = {}
    # extract only directories in the given folder
    subfolders = [name for name in os.listdir(impath) if os.path.isdir(
        os.path.join(impath, name))]
    # if there are no subfolders, walk through files. Take 1 as key for the 
    # categories object
    if not subfolders:
        [_, _, files] = os.walk(os.path.join(impath)).next()
        # this only takes entries that end with '.png'
        entries = {1: 
            [int(cur_file[cur_file.find('_')+1:-4]) for cur_file
            in files if cur_file.endswith('.png')]}
        img_per_cat.update(entries)
        subfolders = ['']
    # if there are subfolders, walk through them
    else:
        for directory in subfolders:
            [_, _, files] = os.walk(os.path.join(impath, directory)).next()
            # this only takes entries that end with '.png'. Strips ending and
            # considers everything after the first '_' as the imagenumber
            imagenumbers = [int(cur_file[cur_file.find('_')+1:-4]) 
                    for cur_file in files
                        if (cur_file.endswith('.png') & (len(cur_file) > 4))]
            entries = {int(directory): imagenumbers}
            img_per_cat.update(entries)
            del directory
    del imagenumbers

    # in case subfolders do not exist, '' is appended here.
    _, features, files = os.walk(os.path.join(ftrpath, 
                                            subfolders[0])).next()
    return Categories(loader, img_per_cat = img_per_cat, features = features)
    
