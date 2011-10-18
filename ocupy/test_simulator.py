#!/usr/bin/env python
# encoding: utf-8


import os
import unittest
import numpy as np
from ocupy import fixmat
import simulator


class TestSimulator(unittest.TestCase):
	
	def test_init(self):
		fm = fixmat.FixmatFactory('/home/student/s/sharst/Dropbox/NBP/fixmat_photos.mat')
		gen = simulator.FixGen(fm)
		self.assertTrue(type(gen.fm)==ocupy.FixMat.fixmat)
		# In this data, the first fixation is deleted from the set
		self.assertTrue(gen.firstfixcentered == True)
		
		fm.fix-=1
		gen = simulator.FixGen(fm)
		self.assertTrue(gen.firstfixcentered == False)
		
	def test_anglendiff(self):
		fm = fixmat.FixmatFactory('/home/student/s/sharst/Dropbox/NBP/fixmat_photos.mat')
		gen = simulator.FixGen(fm)
		coord = [(0,0)]
		
		angle = 45
		length = 1
		cur_angle = angle
		
		for j in range(1000):
			coord.append(gen._calc_xy(coord[-1], cur_angle, length))
			
			if angle < 0:
				cur_angle-=angle
			else:
				cur_angle+=angle
				
		## next step: put coordinates into fixmat
		# fixmat.x = [x[0] for x in coord]
		# put fixmat into anglendiff - result should be 45 and 0
		
		
if __name__ == '__main__':
    unittest.main()