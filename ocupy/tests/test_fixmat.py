#!/usr/bin/env python
# encoding: utf-8

import os
import unittest

from pkgutil import get_data
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.io import loadmat

from ocupy import fixmat, stimuli, loader

import test_loader


class TestFixmat(unittest.TestCase):

    #        Test the interface
    def test_interface(self):   
        fm = fixmat.TestFixmatFactory(categories = [1,2,3], 
                                filenumbers = [1,2,3,4,5,6],
                                subjectindices = [1, 2, 3, 4, 5, 6],
                                params = {'pixels_per_degree':10, 'image_size':[100,500]})
        # We can produce a pretty table with the fixmats parameters by
        # printin it:
        print fm
        # Check parameter access
        self.assertTrue( fm.pixels_per_degree == 10)
        self.assertTrue( fm.image_size[0] == 100 and fm.image_size[1] == 500 )
        
        # Check that all fields can be accessed
        self.assertTrue( (np.unique(fm.SUBJECTINDEX) == np.array([1,2,3,4,5,6])).all() )
        self.assertTrue( (np.unique(fm.category) == np.array([1,2,3])).all() )
        self.assertTrue( (np.unique(fm.filenumber) == np.array([1,2,3,4,5,6])).all() )
        self.assertTrue( len(fm.x) == len(fm.y) and len(fm.y) == len(fm.SUBJECTINDEX) )
        self.assertTrue( len(fm.SUBJECTINDEX) == len(fm.filenumber) )
       
        # Test filtering
        fm_sub1 = fm[fm.SUBJECTINDEX == 1]
        self.assertTrue( (np.unique(fm_sub1.SUBJECTINDEX) == np.array([1])).all() )
        self.assertTrue( (np.unique(fm_sub1.category) == np.array([1,2,3])).all() )
        self.assertTrue( (np.unique(fm_sub1.filenumber) == np.array([1,2,3,4,5,6])).all() )
        self.assertTrue( len(fm_sub1.x) == len(fm_sub1.y) and len(fm_sub1.y) == len(fm_sub1.SUBJECTINDEX) )
        self.assertTrue( len(fm_sub1.SUBJECTINDEX) == len(fm_sub1.filenumber) )
        
        fm_cmp = fm.filter(fm.SUBJECTINDEX == 1)
        for (a,b) in zip(fm_cmp.x, fm_sub1.x): self.assertTrue( a==b )
        
        # Test save and load
        fm_sub1.save('/tmp/test_fixmat')
        fm_cmp = fm.load('/tmp/test_fixmat')
        self.compare_fixmats(fm_sub1, fm_cmp)
       
        self.assertTrue( 'pixels_per_degree' in fm.parameters() )
        self.assertTrue( 'image_size' in fm.parameters() )
 
        # Test iterating over fixmat
        for (img, img_mat) in zip([1,2,3,4,5,6],fm.by_field('filenumber')):
            self.assertEquals( len(np.unique(img_mat.filenumber)), 1 )
            self.assertEquals( np.unique(img_mat.filenumber)[0],  img)
        
        # Test adding fields
        fm.add_field('x2', fm.x)
        for (x1,x2) in zip(fm.x, fm.x2):
            self.assertEquals(x1,  x2)
        
        self.assertRaises(ValueError, lambda: fm.add_field('x3', [1]))
        self.assertRaises(ValueError, lambda: fm.add_field('x2', fm.x))
      
        # Test removing fields:
        fm.rm_field('x2')
        self.assertRaises(ValueError, lambda: fm.rm_field('x2'))
    
        # Add a new subject
        fm_add = fixmat.TestFixmatFactory(categories = [7], filenumbers = [10], subjectindices = [100],
                                    params = {'pixels_per_degree':10,'image_size':[100,500]})
        fm.join(fm_add)
        fm_add2 = fixmat.TestFixmatFactory(categories = [7], filenumbers = [10], subjectindices = [100, 101],
                                    params = {'pixels_per_degree':10,'image_size':[100,500]})
        self.assertRaises(RuntimeError, lambda: fm.join(fm_add2))
        fm_add = fixmat.TestFixmatFactory(categories = [7], filenumbers = [10], subjectindices = [100, 101],
                                    params = {'pixels_per_degree':10,'image_size':[101,500]})
        self.assertRaises(RuntimeError, lambda: fm.join(fm_add2))

        fm_cmp = fm[ (fm.SUBJECTINDEX == 100) & (fm.category == 7) ]
        self.compare_fixmats(fm_add, fm_cmp)
    

    def compare_fixmats(self, a, b):
        for field in a.fieldnames():
            for (v1, v2) in zip(a.__dict__[field], b.__dict__[field]):
                self.assertEquals(v1, v2) 

    def gen_sub(self, subind,numfix):
        fm = fixmat.TestFixmatFactory(subjectindices = [subind],
                 points = [range(0,numfix),range(0,numfix)],
                 categories = [1,2,3,4,5,6,7],
                 filenumbers = [1,2,3,4,5,6,7])
        return fm
    
    def test_getattr(self):
        # Test the 
        # Set up a fake dir structure to generate an aligned fixmat 
        img_per_cat = {1:range(1,11), 2:range(1,11)}
        features = ['a', 'b']
        path, ftrpath = test_loader.create_tmp_structure(img_per_cat, features = features)
        l = loader.LoadFromDisk(impath = path, 
                                ftrpath = ftrpath, 
                                size = (100,100)) 
        inp = stimuli.Categories(l, img_per_cat, features) 
        fm = fixmat.TestFixmatFactory(categories = [1,2], 
                                filenumbers = range(1,11),
                                subjectindices = [1, 2, 3, 4, 5, 6],
                                params = {'pixels_per_degree':10, 'image_size':[100,100]},
                                categories_obj = inp)
        
        fm_err = fixmat.TestFixmatFactory(categories = [1,2], 
                                filenumbers = range(1,11),
                                subjectindices = [1, 2, 3, 4, 5, 6],
                                params = {'pixels_per_degree':10, 'image_size':[100,100]})

        # Now let's check if we can access all the images 
        # and all the features. 
        fm.add_feature_values(['a', 'b'])
        self.assertRaises(RuntimeError, lambda: fm_err.add_feature_values(['a', 'b']))
        for cat_mat, cat_inp in fm.by_cat():
            self.assertEquals(cat_mat.category[0], cat_inp.category)
            for img_mat, img_inp in cat_mat.by_filenumber():
                self.assertEquals(img_mat.filenumber[0], img_inp.image)
        self.assertEquals(len(fm.a), len(fm.x))
        self.assertEquals(len(fm.b), len(fm.x))
        # Let's also check if make_reg_data works
        a, b = fm.make_reg_data(features)
        self.assertEquals(a.shape[1], len(fm.x))
        self.assertEquals(a.shape[0], len(features))
        self.assertEquals(b.shape[1], len(fm.x))
        self.assertEquals(b.shape[0], len(features))
        self.assertEquals(b.sum(), a.sum())
        a, b = fm.make_reg_data(features, all_controls = True)
        self.assertEquals(a.shape[1], len(fm.x))
        self.assertEquals(a.shape[0], len(features))
        self.assertEquals(b.shape[1], len(fm.x))
        self.assertEquals(b.shape[0], len(features))
        self.assertEquals(b.sum(), a.sum())
        test_loader.rm_tmp_structure(path)
        test_loader.rm_tmp_structure(ftrpath) 
 
    def test_single(self):
        numfix = 1
        fm = self.gen_sub(1,numfix)
        for (cat, cat_mat) in enumerate(fm.by_field('category')):
            for (img, img_mat) in enumerate(cat_mat.by_field('filenumber')):
                self.assertEquals(len(img_mat.x), 1)
                self.assertEquals(img_mat.SUBJECTINDEX[0], 1)
                self.assertEquals(img_mat.filenumber[0], img+1)
                self.assertEquals(img_mat.category[0], cat+1)
    
    def test_multiple_subs(self):
        import random
        numfix  = random.randint(10,100)
        numsubs = random.randint(10,20)
        fm_all = self.gen_sub(0,numfix)
        for i in range(1,numsubs):
            fm_all.join(self.gen_sub(i,numfix))
        for (i,sub_mat) in enumerate(fm_all.by_field('SUBJECTINDEX')):
            for (cat,cat_mat) in enumerate(sub_mat.by_field('category')):
                for (img,img_mat) in enumerate(cat_mat.by_field('filenumber')):
                    self.assertEquals(len(img_mat.x), numfix)
                    self.assertEquals(img_mat.SUBJECTINDEX[0], i)
                    self.assertEquals(img_mat.filenumber[0], img+1)
                    self.assertEquals(img_mat.category[0], cat+1)
    
    def test_attribute_access(self):
        fm = self.gen_sub(0,100) 
        fsi = fm.SUBJECTINDEX
        fx = fm.x
        fy = fm.y
        for k in range(0,100):
            self.assertEquals(fsi[k], 0)
            self.assertEquals(fx[k], k)
            self.assertEquals(fy[k], k)
   
    def test_factories(self):
        with NamedTemporaryFile(mode = 'w', prefix = 'fix_occ_test',
                suffix = '.mat') as ntf:
            ntf.write(get_data('ocupy.tests', 'fixmat_demo.mat'))
            ntf.seek(0)
            fm  = fixmat.FixmatFactory(ntf.name)
        with NamedTemporaryFile(mode = 'w', prefix = 'fix_occ_test',
                suffix = '.mat') as ntf:
            ntf.write(get_data('ocupy.tests', 'fixmat_demo.mat'))
            ntf.seek(0)
            fm2 = fixmat.DirectoryFixmatFactory(os.path.dirname(ntf.name), glob_str = 'fix_occ_test*.mat' )
            self.compare_fixmats(fm, fm2)   
            self.assertRaises(ValueError, lambda: fixmat.DirectoryFixmatFactory('.', glob_str = 'xxx*.mat' ))
         
    def test_cmp2fixmat(self):
        # This test only works with some scipy versions. 
        with NamedTemporaryFile(mode = 'w', prefix = 'fix_occ_test',
                suffix = '.mat') as ntf:
            ntf.write(get_data('ocupy.tests', 'fixmat_demo.mat'))
            ntf.seek(0)
            fm  = fixmat.FixmatFactory(ntf.name)
        with NamedTemporaryFile(mode = 'w', prefix = 'fix_occ_test',
                suffix = '.mat') as ntf:
            ntf.write(get_data('ocupy.tests', 'fixmat_demo.mat'))
            ntf.seek(0)
            fm_ref = loadmat(ntf.name, struct_as_record=True)['fixmat'][0][0]
        for field in fm._fields:
            l1 = fm_ref[field]
            l2 = fm.__dict__[field]
            self.assertEquals(l1.size, l2.size)
            self.assertTrue((l1 == l2).all())
               
if __name__ == '__main__':
    unittest.main()
