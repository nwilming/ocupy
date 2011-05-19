#!/usr/bin/env python

import os
from os.path import join
import unittest
from tempfile import mkdtemp

from scipy.io import savemat
import Image
import numpy as np

from ocupy import loader


class TestLoader(unittest.TestCase):

    #Test the interface
    def test_load_from_disk(self):
        # Generate a tmp fake data structure
        img_per_cat = {2:range(1,10), 8:range(30,40), 111:range(6,15)}
        path,_ = create_tmp_structure(img_per_cat)        
        l = loader.LoadFromDisk(impath = path)
        for cat in img_per_cat.keys():
            for image in img_per_cat[cat]:
                l.get_image(cat,image)
                self.assertTrue(l.test_for_category(cat))
                self.assertTrue(l.test_for_image(cat, image))
        # Test checks for non existing images
        no_img_per_cat = {0:range(1,10), 7:range(30,40), 110:range(6,15)}
        for cat in no_img_per_cat.keys():
            for image in no_img_per_cat[cat]:
                self.assertTrue(not l.test_for_category(cat))
                self.assertTrue(not l.test_for_image(cat, image))
        no_img_per_cat = {2:range(11,20), 8:range(20,30), 111:range(16,40)}
        for cat in no_img_per_cat.keys():
            for image in no_img_per_cat[cat]:
                self.assertTrue(not l.test_for_image(cat, image)) 
        rm_tmp_structure(path)
 
    def test_load_from_disk_scaling(self):
        # Generate a tmp fake data structure
        img_per_cat = {2:range(1,10), 8:range(30,40), 111:range(6,15)}
        path,_ = create_tmp_structure(img_per_cat) 
        l = loader.LoadFromDisk(impath = path, size = (10,10))
        for cat in img_per_cat.keys():
            for image in img_per_cat[cat]:
                im = l.get_image(cat,image)
                self.assertTrue(im.shape[0] == 10 and im.shape[1] == 10)
                self.assertTrue(l.test_for_category(cat))
                self.assertTrue(l.test_for_image(cat, image))
        rm_tmp_structure(path)

    def test_save_to_disk(self):
        path = mkdtemp()
        ftrpath = mkdtemp()
        im_tmp = np.ones((100,100))
        l = loader.SaveToDisk(impath = path,ftrpath = ftrpath, size = (10,10))
        # Generate a tmp fake data structure
        img_per_cat = {2:range(1,10), 8:range(30,40), 111:range(6,15)}
        for cat in img_per_cat.keys():
            # Create category dir
            for image in img_per_cat[cat]:
                l.save_image(cat, image, im_tmp)
                for f in ['a', 'b']:
                    l.save_feature(cat, image, f, np.ones((10,10))) 
        l = loader.LoadFromDisk(impath = path, ftrpath = ftrpath, size = (10,10))
        for cat in img_per_cat.keys():
            for image in img_per_cat[cat]:
                im = l.get_image(cat,image)
                self.assertTrue(im.shape[0] == 10 and im.shape[1] == 10)
                self.assertTrue(l.test_for_category(cat))
                self.assertTrue(l.test_for_image(cat, image))
                self.assertTrue(l.test_for_feature(cat, image, 'a'), True) 
        os.system('rm -rf %s' %path)
        os.system('rm -rf %s' %ftrpath)

    
    def test_testloader(self):
        img_per_cat = {1: range(1,10), 2: range(1,10)}
        l = loader.TestLoader(img_per_cat = img_per_cat, features = ['a', 'b'])
        for cat in range(1,10):
            for img in range(1,100):
                if cat in img_per_cat.keys() and img in img_per_cat[cat]: 
                    self.assertEquals(l.test_for_category(cat), True)
                    self.assertEquals(l.test_for_image(cat, img), True)
                    self.assertEquals(l.test_for_feature(cat, img, 'a'), True)
                    l.get_image(cat, img)
                    l.get_feature(cat, img, 'a')
                    l.get_feature(cat, img, 'b')
                    self.assertTrue(l.test_for_feature(cat, img, 'a'))
                else:
                    self.assertEquals(l.test_for_image(cat, img), False)
                    self.assertRaises(IndexError, lambda: l.get_image(cat, img))
 
def create_tmp_structure(img_per_cat, features = None):
    im_tmp = Image.fromarray(np.ones((1,1))).convert('RGB')   
    path = mkdtemp()
    ftr_path = None
    if features:
        ftr_path = mkdtemp()
    for cat in img_per_cat.keys():
        # Create category dir
        os.mkdir(join(path, str(cat)))
        if features:
            for feature in features:
                os.makedirs(join(ftr_path, str(cat), str(feature))) 
        for image in img_per_cat[cat]:
            im_tmp.save(join(path,str(cat),'%i_%i.png'%(cat, image)))
            if features:
                for feature in features:
                    savemat(join(ftr_path,str(cat),
                            feature, '%i_%i.mat'%(cat, image)),
                            {'output':np.ones((1,1))})
    return path, ftr_path

def rm_tmp_structure(path):
    os.system('rm -rf %s' %path)

    

if __name__ == '__main__':
    unittest.main()
