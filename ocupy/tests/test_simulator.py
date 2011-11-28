#!/usr/bin/env python
# encoding: utf-8

import pdb

import unittest
import numpy as np
from ocupy import fixmat
import ocupy
import simulator


class TestSimulator(unittest.TestCase):
    
    def test_init(self):
        fm = fixmat.FixmatFactory('/home/student/s/sharst/Dropbox/NBP/fixmat_photos.mat')
        gen = simulator.FixGen(fm)
        self.assertTrue(type(gen.fm)==ocupy.fixmat.FixMat)
        # In this data, the first fixation is deleted from the set
        self.assertTrue(gen.firstfixcentered == True)
        
        fm.fix-=1
        gen = simulator.FixGen(fm)
        self.assertTrue(gen.firstfixcentered == False)
        
    def test_anglendiff(self):
        fm = fixmat.FixmatFactory('/home/student/s/sharst/Dropbox/NBP/fixmat_photos.mat')
        gen = simulator.FixGen(fm)
        
        for angle in [-180, -90, 0, 45, 90, 135]:
            print "Running anglendiff test on "+repr(angle)+" degrees."
            coord = [(0,0)]
            length = 1
            cur_angle = [np.nan]
            cur_angle.append(angle)
        
            # Create artificial fixmat with fixed angle differences
            for j in range(len(fm.x)-1):
                coord.append(gen._calc_xy(coord[-1], cur_angle[-1], length))
                cur_angle.append(cur_angle[-1]+angle)
                
            fm.x = np.array([x[0] for x in coord])
            fm.y = np.array([x[1] for x in coord])
            
            gen.initializeData()
            
            # Use anglendiff to calculate angles and angle_diff
            a, l, ad, ld = simulator.anglendiff(fm, return_abs=True)
            a = np.round(a[0])  
            a[a==-180]=180
            
            cur_angle = simulator.reshift(np.array(cur_angle[0:-1]))
            
            # Assign nans to make arrays comparable
            cur_angle[np.isnan(a)]=np.nan
            cur_angle[np.round(cur_angle)==-180]=180
                
            ad = np.round(simulator.reshift(ad[0][~np.isnan(ad[0])]))
            pdb.set_trace()
            
            if (angle==180 or angle==-180):
                self.assertTrue(np.logical_or(ad==angle, ad==-angle).all())
            
            else:
                self.assertTrue((ad==angle).all())
                self.assertTrue((a[~np.isnan(a)]==cur_angle[~np.isnan(cur_angle)]).all())
    
    
    def test_simulating(self):
    
        fixhists = []
        # Create some fixhists for testing
        
        #Crosses
        K=np.eye(73,73)+np.rot90(np.eye(73,73))
        K = np.concatenate((K,K,K,K,K),axis=1)[0:,0:-5]
        fixhists.append(K/K.sum())
        
        #Rectangle
        K=np.zeros((73,360))
        K[0:3]=1
        K[-2:]=1
        K[0:,1:3]=1
        K[0:,-3:-1]=1
        fixhists.append(K/K.sum())
        
        # Stripe
        K = np.zeros((73,360))
        K[:,35:37]=1
        
        
        
    
        
if __name__ == '__main__':
    unittest.main()
