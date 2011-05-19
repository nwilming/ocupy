#!/usr/bin/env python
"""
A cross-validation interface and an implementation of a simple 
cross-validation scheme.
"""

import numpy as np

from ocupy.utils import randsample, ismember
from ocupy.stimuli import Categories


class XValidation(object):
    """
    Interface for a cross-validation object.
    """
    def __init__(self, categories, fixmat, num_slices):
        """Initializes xvalidation object with categories and fixmat."""
        raise NotImplementedError

    def generate(self, subject_partition, image_partition):
        """Generates num_slices of data where subject_partition define how 
        many subjects go into test and training set and image_partition does
        the same for images."""
        raise NotImplementedError        


class SimpleXValidation(XValidation):
    """
    SimpleXValidation performs a simple hold-out cross-validation.
    """
    def __init__(self, fm, categories, 
                    subject_hold_out = .3, 
                    image_hold_out=.3,
                    num_slices = 10):
        self.fm = fm
        self.num_slices = num_slices
        self.categories = categories
        self.subject_hold_out = subject_hold_out
        self.image_hold_out = image_hold_out
        
    def generate(self): 
        """
        Generator for creating the cross-validation slices.

        Returns:
            A tuple of that contains two fixmats (training and test)
            and two Category objects (test and train).
        """
        for _ in range(0, self.num_slices): 
            #1. separate fixmat into test and training fixmat
            subjects = np.unique(self.fm.SUBJECTINDEX)
            test_subs  = randsample(subjects,
                            self.subject_hold_out*len(subjects))  
            train_subs = [x for x in subjects if x not in test_subs] 
            test_fm  = self.fm[ismember(self.fm.SUBJECTINDEX, test_subs)]
            train_fm = self.fm[ismember(self.fm.SUBJECTINDEX, train_subs)]
            
            #2. distribute images 
            test_imgs = {}
            train_imgs = {}
            id_test = (test_fm.x <1) & False
            id_train = (train_fm.x <1)  & False
            for cat in self.categories:
                imgs = cat.images()
                test_imgs.update({cat.category:randsample(imgs,
                                self.image_hold_out*len(imgs)).tolist()})
                train_imgs.update({cat.category:[x for x in imgs 
                                if not ismember(x, test_imgs[cat.category])]})
                id_test = id_test | ((ismember(test_fm.filenumber, 
                                test_imgs[cat.category])) & 
                                (test_fm.category == cat.category))
                id_train = id_train | ((ismember(train_fm.filenumber, 
                                train_imgs[cat.category])) & 
                                (train_fm.category == cat.category))


            #3. Create categories objects and yield result
            test_stimuli = Categories(self.categories.loader, test_imgs,
                                      features=self.categories._features,
                                      fixations=test_fm)
            train_stimuli = Categories(self.categories.loader, train_imgs,
                                       features=self.categories._features,
                                       fixations=train_fm)
            yield (train_fm[id_train], 
                   train_stimuli, 
                   test_fm[id_test], 
                   test_stimuli)

 
