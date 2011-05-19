#!/usr/bin/env python
# encoding: utf-8

import unittest
import numpy as np

from ocupy import fixmat, bounds


class TestBounds(unittest.TestCase):
    def setUp(self):
        self.fm = fixmat.TestFixmatFactory(categories = [1,2,3], 
                                filenumbers = [1,2,3,4,5,6],
                                subjectindices = [1, 2, 3, 4, 5, 6],
                                params = {'pixels_per_degree':1, 'image_size':[100,500]})
        
    # test for category > 0 assertion
    def test_intersubject_scores(self):
        auc, kl, nss = bounds.intersubject_scores(self.fm,
                1, [1,2,3,4], [1, 2, 3, 4], [5, 6], [5, 6])
        self.assertTrue(auc > 0.99)
        auc, kl, nss = bounds.intersubject_scores(self.fm,
                1, [1,2,3,4], [1, 2, 3, 4], [5, 6], [5, 6],
                controls = True)
        self.assertTrue(auc > 0.99)
        auc, kl, nss = bounds.intersubject_scores(self.fm,
                1, [7], [7], [5, 6], [5, 6],
                controls = True)
        self.assertTrue(np.isnan(auc) and np.isnan(kl) and np.isnan(nss))

    def test_intersubject_scores_2(self):
        auc, kl, nss = bounds.intersubject_scores_random_subjects(self.fm,
                1, 1, 5, 1)
        self.assertTrue(auc > 0.99)
        self.assertRaises(ValueError, lambda : bounds.intersubject_scores_random_subjects(self.fm,
                    1, 1, 6, 1))
        self.assertRaises(ValueError,  lambda : bounds.intersubject_scores_random_subjects(self.fm,
                    1, 6, 1, 6))
        auc, kl, nss = bounds.intersubject_scores_random_subjects(self.fm,
                    1, 100, 1, 5)
        self.assertTrue(np.isnan(auc) and np.isnan(kl) and np.isnan(nss))
 
    def check_bounds(self, auc):
        self.assertEquals(len(auc.keys()), 3)
        for cat in np.unique(self.fm.category):
            self.assertEqual(len(auc[cat]), 6)
            for val in auc[cat]:
                self.assertTrue(val > 0.99)             

   
    def test_upper_bound(self):
        # The test fixmat has no variance, so the upper bound
        # should be really high.
        auc, kl, nss = bounds.upper_bound(self.fm)
        self.check_bounds(auc)
        auc, kl, nss = bounds.upper_bound(self.fm, 3)
        self.check_bounds(auc)
        self.assertRaises(AssertionError, lambda: bounds.upper_bound(self.fm, 100))

    def test_lower_bound(self):
        # The test fixmat has no variance, so the lower bound
        # should be really high.
        auc, kl, nss = bounds.lower_bound(self.fm)
        self.check_bounds(auc)
        auc, kl, nss = bounds.lower_bound(self.fm, nr_subs = 3)
        self.check_bounds(auc)
        auc, kl, nss = bounds.lower_bound(self.fm, nr_imgs = 3)
        self.check_bounds(auc)
        auc, kl, nss = bounds.lower_bound(self.fm, nr_subs = 3, nr_imgs = 3)
        self.check_bounds(auc)
        self.assertRaises(AssertionError, lambda: bounds.lower_bound(self.fm, nr_imgs = 100))
        self.assertRaises(AssertionError, lambda: bounds.lower_bound(self.fm, nr_subs = 100))

    def tearDown(self):
        self.fm = None
        
if __name__ == '__main__':
    unittest.main()
