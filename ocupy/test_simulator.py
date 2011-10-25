#!/usr/bin/env python
# encoding: utf-8

import pdb

import os
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
		
		for angle in [-90, 0, 45, 90, 135]:
			print "Running anglendiff test on "+repr(angle)+" degrees."
			coord = [(0,0)]
	
			length = 1
			cur_angle = [np.nan]
			cur_angle.append(angle)
		
			for j in range(len(fm.x)-1):
				coord.append(gen._calc_xy(coord[-1], cur_angle[-1], length))
				cur_angle.append(cur_angle[-1]+angle)
				
			fm.x = np.array([x[0] for x in coord])
			fm.y = np.array([x[1] for x in coord])
			
			a, l, ad, ld = simulator.anglendiff(fm, return_abs=True)
			a = a[0]
			
			cur_angle = np.array(cur_angle)
			# Assign nans to make arrays comparable
			cur_angle[np.isnan(a)]=np.nan
			cur_angle = simulator.reshift(cur_angle[0:-1]) # One element is added too much
			cur_angle[cur_angle==-180]=180
			#test = a==-180
			
			## Problem: Doesnt work!
			
			
			#print a[test][0:20]
			#a[test]=180
			#print a[test][0:20]
			a[a==-180]=180
			angles = a
			
			ad[0] = np.round(simulator.reshift(ad[0][~np.isnan(ad[0])]))
			
			if (angle==180 or angle==-180):
				self.assertTrue(np.logical_or(ad[0]==angle, ad[0]==-angle).all())
				
			else:
				self.assertTrue((ad[0]==angle).all())
				pdb.set_trace()
				self.assertTrue((np.round(a[~np.isnan(a)])==simulator.reshift(cur_angle[~np.isnan(cur_angle)])).all())
				

	
		
if __name__ == '__main__':
    unittest.main()