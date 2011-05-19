#!/usr/bin/env python
# encoding: utf-8

import os
from pkgutil import get_data
import unittest
from tempfile import NamedTemporaryFile

from ocupy import fixmat, stimuli, loader

import test_loader

class TestStimuli(unittest.TestCase):
    
    def setUp(self):
        self.features = range(1,11)
        self.inputs = {'a':{1:self.features,2:self.features,3:self.features},'b':{10:self.features}}
        self.test_loader = loader.TestLoader(self.inputs)
        self.inp = stimuli.Categories(self.test_loader,self.inputs)          

    
    def test_directory_stimuli_factory(self):
        img_per_cat = {1:range(1,11), 2:range(1,11)}
        features = ['a', 'b']
        path, ftrpath = test_loader.create_tmp_structure(img_per_cat, features = features)
        l = loader.LoadFromDisk(impath = path, 
                                ftrpath = ftrpath, 
                                size = (100,100)) 
        inp = stimuli.DirectoryStimuliFactory(l)
        for cat in img_per_cat.keys():
            inp[cat]
            for img in img_per_cat[cat]:
                inp[cat][img]
                inp[cat][img]['a']
                inp[cat][img].data
 
        test_loader.rm_tmp_structure(path)
        test_loader.rm_tmp_structure(ftrpath) 

    
    def test_load_phantom_input(self):    
        self.assertRaises(IndexError, lambda : self.inp['c'])
        self.assertRaises(IndexError, lambda : self.inp['a'][5])
        self.assertRaises(IndexError, lambda : self.inp['a'][1][55])


    def test_fixations(self):
        img_per_cat = {7:range(1,65),8:range(1,65)}
        l = loader.TestLoader(img_per_cat, size = (10,10))
        with NamedTemporaryFile(mode = 'w', prefix = 'fix_occ_test',
                suffix = '.mat') as ntf:
            ntf.write(get_data('ocupy.tests', 'fixmat_demo.mat'))
            ntf.seek(0)
            fm = fixmat.FixmatFactory(ntf.name)
        fm = fm[fm.category>0]
        inp = stimuli.FixmatStimuliFactory(fm,l)
        # Now we can iterate over the input object and get fixations on each image
        for cat in inp:
            for img in cat:
                self.assertEqual(img.fixations.filenumber[0], img.image)
                self.assertEqual(img.fixations.category[0], img.category)
        inp = stimuli.Categories(l, img_per_cat)
        self.assertRaises(RuntimeError, lambda : inp.fixations)
        self.assertRaises(RuntimeError, lambda : inp[7].fixations)
        self.assertRaises(RuntimeError, lambda : inp[7][1].fixations)
   
    def test_general(self):
        #Create an input object: There are two ways, one is the 
        features = ['a','b','c','d']
        img_per_cat = {7:range(1,65),8:range(1,65)}
        l = loader.TestLoader(img_per_cat,features, size = (10,10))
        inp = stimuli.Categories(l, img_per_cat, features)
        # In this case it should have two categories (2,9) with 10 and 50 images
        # Let's check this
        self.assert_(7 in inp.categories())
        self.assert_(8 in inp.categories())
        self.assertEquals(len(inp[7].images()), 64)
        self.assertEquals(len(inp[8].images()), 64)
        # Good, now we can access some data
        img = inp[7][16].data
        self.assertTrue(img.shape[0] == 10 and img.shape[1] == 10)
        # In general we can use [] to access elements in the object
        # But we can also iterate trough these objects
        # This should take a long time because it loads all images 
        # from disk
        for cat in inp:
            for img in cat:
                img.image
                img.category
                img.data

    def test_fixmat_input_factory(self): 
        # Another way to create an input object is with the
        # FixmatInputFactory 
        features = ['a','b','c','d']
        img_per_cat = {7:range(1,65),8:range(1,65)}
        l = loader.TestLoader(img_per_cat,features, size = (10,10))
        with NamedTemporaryFile(mode = 'w', prefix = 'fix_occ_test',
                suffix = '.mat') as ntf:
            ntf.write(get_data('ocupy.tests', 'fixmat_demo.mat'))
            ntf.seek(0)
            fm = fixmat.FixmatFactory(ntf.name)
            fm = fm[fm.category>0]
            inp = stimuli.FixmatStimuliFactory(fm,l)
        # Now we can iterate over the input object and get fixations on each image
        for cat in inp:
            for img in cat:
                self.assertEquals(img.fixations.filenumber[0], img.image)
                self.assertEquals(img.fixations.category[0], img.category)


    def compare_fixmats(self, a,b):
        for field in a.fieldnames():
            for (v1, v2) in zip(a.__dict__[field], b.__dict__[field]):
                self.assertEquals(v1, v2) 

    
         
if __name__ == '__main__':
    unittest.main()
