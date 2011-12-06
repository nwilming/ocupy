#!/usr/bin/env python
# encoding: utf-8

import os
from os.path import join
import unittest

import numpy as np
import scipy.ndimage as sndi
from scipy.stats.kde import gaussian_kde

from ocupy import fixmat, simulator, spline_base as sb

from scikits.learn import linear_model
from ocupy import measures
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
    
    def skip_cosine_fit(self):
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
    
    def test_compare_3d(self):
        width, height, depth = 15,15,15
        mat = 0.1*np.random.random((width, height, depth))
        idx = range(0, np.prod(mat.shape))
        mat[1:5,:,3] +=5
        np.random.shuffle(idx)
        idx = list(idx)
        mat = mat.flatten()[idx].reshape((width, height, depth))
        pdist = sndi.gaussian_filter(mat,3)
        pdist += min(pdist.flatten())
        pdist = pdist/sum(pdist)

        cdist = cumsum(pdist.flat)
        kl_fit = []
        kl_kde = []
        kl_h = []
        xs = []
        ion()
        for num_samples in np.logspace(2,3.5,10):
            xs += [num_samples]
            print num_samples
            samples = []
            for x in range(int(num_samples)):
                s = simulator.drawFrom(cdist)
                x,y,z = np.unravel_index(s, pdist.shape)
                samples += [[x,y,z]]
            samples = np.array(samples)

            e_y = e_x = linspace(0,width,width+1)
            e_z = linspace(0,depth,depth+1)
            fit, h = sb.fit3d(samples, e_y, e_x, e_z)
            k = fit<0
            fit[k] = 0   
            fit = fit/sum(fit.flat)
            h = h/sum(h.flat)
            kde = gaussian_kde(samples.astype(float).T)
            x, y, z = mgrid[0:width,0:height, 0:depth]
            kde_fit = kde.evaluate((x.flatten(), y.flatten(), z.flatten())).reshape((width,height,depth))
            kde_fit = kde_fit + min(kde_fit.flat)
            kde_fit/=sum(kde_fit.flat)
            kl_fit += [measures.kldiv(fit,pdist)]
            kl_h += [measures.kldiv(h,pdist)]
            kl_kde += [measures.kldiv(kde_fit,pdist)]


            print 'O vs. F: ', measures.kldiv(pdist, fit)
            print 'O vs. K: ', measures.kldiv(pdist, kde_fit)
            print 'O vs. H:', measures.kldiv(pdist, h)
            kde_fit = kde_fit.sum(2)
            fit = fit.sum(2)
            h = h.sum(2)
            pdist2 = pdist.sum(2)
            mx = np.max([np.max(h), np.max(kde_fit.flatten()), np.max(fit.flatten()), np.max(pdist2.flatten())])
            mi = np.min([np.min(h), np.min(kde_fit.flatten()), np.min(fit.flatten()), np.min(pdist2.flatten())])
            subplot(1,5,1)
            plot(xs,kl_fit)
            #plot(xs,kl_h)
            plot(xs,kl_kde)
            subplot(1,5,2)
            imshow(fit)
            clim([mi,mx])
            subplot(1,5,3)
            imshow(kde_fit)
            clim([mi,mx])
            subplot(1,5,4)            
            imshow(h)
            clim([mi,mx])
            subplot(1,5,5)
            imshow(pdist2)
            clim([mi,mx])

            waitforbuttonpress()
        legend(['Spline fit', 'KDE fit'])
        
        show()
        
         
            
    def skip_compare_2d(self):
        mat = 2*np.eye(100) +  np.random.random((100,100))
        np.random.shuffle(mat)
        pdist = sndi.gaussian_filter(mat,5)
        pdist = pdist/sum(pdist)
        cdist = cumsum(pdist.flat)
        kl_fit = []
        kl_kde = []
        kl_h = []
        xs = []
        ion()
        for num_samples in np.logspace(2,3.5,10):
            xs += [num_samples]
            print num_samples
            samples = []
            for x in range(int(num_samples)):
                s = simulator.drawFrom(cdist)
                x,y = np.unravel_index(s, pdist.shape)
                samples += [[x,y]]
            samples = np.array(samples)
            
            e_y = e_x = linspace(0,100,101)
            fit, h = sb.fit2d(samples, e_y, e_x)
            k = fit<0
            fit[k] = 0   
            fit = fit/sum(fit.flat)
            
            kde = gaussian_kde(samples.astype(float).T)
            x, y = mgrid[0:100,0:100]
            kde_fit = kde.evaluate((x.flatten(), y.flatten())).reshape((100,100))
            kde_fit = kde_fit + min(kde_fit.flat)
            kde_fit/=sum(kde_fit.flat)
            kl_fit += [measures.kldiv(fit,pdist)]
            kl_h += [measures.kldiv(h,pdist)]
            kl_kde += [measures.kldiv(kde_fit,pdist)]
            print 'O vs. F: ', measures.kldiv(pdist, fit)
            print 'O vs. K: ', measures.kldiv(pdist, kde_fit)
            print 'O vs. H:', measures.kldiv(pdist, h)
            subplot(1,5,1)
            plot(xs,kl_fit)
            plot(xs,kl_h)
            plot(xs,kl_kde)
            subplot(1,5,2)
            imshow(fit)
            subplot(1,5,3)
            imshow(kde_fit)
            subplot(1,5,4)
            imshow(h)
            subplot(1,5,5)
            imshow(pdist)
            waitforbuttonpress()
        legend(['Spline fit', 'KDE fit', 'Histogram'])
        
        show()
        
            

class TestFixationFits(unittest.TestCase):
    
    def skip_fixmats_1D(self):
        path = '/Users/nwilming/ior/data'
        
        fixmats = [fixmat.FixmatFactory(join(path,fname)) for fname in ['afc_patch_recognition_fixmat.mat',
                    'age_study_fixmat.mat',
                    'patch_recognition_fixmat.mat']]
        for fm in fixmats:
            data = simulator.anglendiff(fm, return_abs=True)
            for i,(datum, name, e) in enumerate(zip(data,
                                            ['Angles', 'Length', 'DAngles', 'DLengths'],
                                            [linspace(0,180,90), linspace(0,700,100),
                                             linspace(-180,180,90),linspace(-700,700,100)])):
                subplot(2,2,i+1)  
                p, h = sb.fit1d(datum[0][0::2], e)
                p2, h2 = sb.fit1d(datum[0][1::2], e)
                plot(p)  
                plot(h2)
                title(name)
        show()

    def skip_fixmats_2D(self):
        path = '/Users/nwilming/ior/data'

        fixmats = [fixmat.FixmatFactory(join(path,fname)) for fname in ['afc_patch_recognition_fixmat.mat',
                    'age_study_fixmat.mat',
                    'patch_recognition_fixmat.mat']]
        cnt = 1
        for fmi, fm in enumerate(fixmats):
            data = simulator.anglendiff(fm, return_abs=True)
            data = [(data[0], data[1]),(data[2], data[3])]
            for i,(datum, name, e) in enumerate(zip(data,
                                            ['Angles and Lengths', 'Delta Angles and Lengths'],
                                            [(linspace(0,180,90), linspace(0,700,100)),
                                             (linspace(-180,180,90),linspace(-700,700,100))])):

                subplot(3,4,cnt)
                cnt+=1
                p, h = sb.fit2d((datum[1][0][0::2],datum[0][0][0::2]) , e[1], e[0])
                p2, h2 = sb.fit2d((datum[1][0][1::2],datum[0][0][1::2]) , e[1], e[0])
                imshow(p)  
                title(name)
                subplot(3,4,cnt)
                cnt+=1
                imshow(h2)
                title(name)
        show()            
    

    
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
