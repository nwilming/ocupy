#!/usr/bin/env python
# encoding: utf-8

import os
import unittest

import numpy as np

from ocupy import spline_base as sb

from scikits.learn import linear_model

import h5py

from pylab import *

class Test1DSplines(unittest.TestCase):

    def skip_cosine_fit(self):
        x = np.cos(np.linspace(0,np.pi*2, 99))
        noise = (np.random.random((99,))-.5)
        plot(x+noise)
        plot(x)
        for num_splines in range(5,20,2):
            splines = sb.spline_base1d(99, num_splines, spline_order = 5)
            regression_models = [linear_model.Ridge(), linear_model.BayesianRidge()]
            names = ['Ridge', 'BayesRidge']
            for name, model in zip(names, regression_models):
                model.fit(splines, x+noise)
                y = model.predict(splines)
                plot(y)
                print num_splines, ' r^2 for ', name, ' is ', np.corrcoef(x+noise, y)[0,1]**2                     
        show()
        
    def skip_tangens(self):
        x = np.tan(np.linspace(0,np.pi, 99))
        noise = (np.random.random((99,))-.5)
        plot(x+noise)
        plot(x)
        for num_splines in range(10,101,10):
            splines = sb.spline_base1d(99, num_splines, spline_order = 5, marginal=x)
            regression_models = [linear_model.Ridge(), linear_model.BayesianRidge()]
            names = ['Ridge', 'BayesRidge']
            for name, model in zip(names, regression_models):
                model.fit(splines, x+noise)
                y = model.predict(splines)
                plot(y)
                print num_splines, ' r^2 for ', name, ' is ', np.corrcoef(x+noise, y)[0,1]**2                     
        show()
    

class Test2DSplines(unittest.TestCase):
    
    def test_cosine_fit(self):
        x = np.cos(np.linspace(0,np.pi*2, 99))
        X = np.dot(x[:,newaxis],x[:,newaxis].T)
        noise = (np.random.random(X.shape)-.5)*0
        cnt = 1
        subplot(4,5,cnt)
        imshow(X+noise)
        for num_splines in range(2,20,2):
            splines = sb.spline_base(99,99, num_splines, spline_order = 3)
            regression_models = [linear_model.Ridge(), linear_model.BayesianRidge()]
            names = ['Ridge', 'BayesRidge']
            for name, model in zip(names, regression_models):
                model.fit(splines.T, (X+noise).flat)
                y = model.predict(splines.T)
                cnt += 1
                subplot(4,5,cnt)
                imshow(y.reshape(X.shape))
                
                print num_splines, ' r^2 for ', name, ' is ', np.corrcoef(X+noise, y.reshape(X.shape))[0,1]**2
        show()
    
    def skip_non_continuous(self):
        x = linspace(-10, 10, 99)
        x = x**2
        X = np.dot(x[:,newaxis],x[:,newaxis].T)
        noise = (np.random.random(X.shape)-.5)
        cnt = 1
        subplot(4,5,cnt)
        imshow(X+noise)
        for num_splines in range(2,20,10):
            splines = sb.spline_base(99,99, num_splines, spline_order = 3)
            regression_models = [linear_model.Ridge(), linear_model.BayesianRidge()]
            names = ['Ridge', 'BayesRidge']
            for name, model in zip(names, regression_models):
                model.fit(splines.T, (X+noise).flat)
                y = model.predict(splines.T)
                cnt += 1
                subplot(4,5,cnt)
                imshow(y.reshape(X.shape))

                print num_splines, ' r^2 for ', name, ' is ', np.corrcoef(X+noise, y.reshape(X.shape))[0,1]**2
        show()

class TestFixationFits(unittest.TestCase):

    def skip_err_samples(self):
        # For now hard code path
        data = h5py.File('/home/student/n/nwilming/Desktop/spline_fit_example.hdf5')
        samples = np.array(data['samples']) 
        e_x = np.linspace(-180.5,179.5,361)
        e_y = np.linspace(-36.5,36.5,74)
        H = sb.spline_pdf(np.array(samples), 
            e_y, e_x, nr_knots_y = 20, nr_knots_x = 20)       
        imshow(H)
        show()


if __name__ == '__main__':
    unittest.main()
