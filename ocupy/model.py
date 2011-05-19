#!/usr/bin/env python
"""This module defines an interface for models."""

class Model(object):
    """
    Abstract base class for models.

    This is a minimalistic interface, that offers an interface for
    training, predicting, loading and saving for models. This suffices
    to carry out a full evaluation of different models. This can be done
    in combination with the cross-validation module.
    """

    def __init__(self, name):
        """
        Every model is required to have a name
        """
        self.name = name
    
    def train(self, fixmat, categories):
        """
        Trains a model / fit parameters of the model.

        Input:
            fixmat: FixMat
                Fixation data that can be used for training the model.
            categories: Categories object
                Stimuli (which are aligned to the fixation data) that can be
                used for training the model.
        
        After this method has been called, the model is in a state in which it
        can generate predictions - nothing has to be returned by this
        function.
        """
        raise NotImplementedError
        
    def predict(self, test_stim, predicted_stims):
        """
        Generates predictions for all elements of test_stim and puts them
        into predicted_stims

        Input:
            test_stim: Categories object
                The stimuli to generate predictions for.
            predicted_stims: Categories object
                A categories object that is writable. The predict method
                iterates over all categories and images in test_stim and puts
                the generated predictions as a new feature (called
                'prediction') into this categories object.
        Raises:
            PredictionError: A prediction error is raised if for some reason a
                prediction can not be generated.
            
        """
        raise NotImplementedError

    def save(self, path):
        """
        Saves model to path such that loading it can restore the state of the
        model.
        """
        raise NotImplementedError
        
    def load(self, path):
        """
        Loads model from path and restores the state of the saved model.
        """
        raise NotImplementedError



 
 
