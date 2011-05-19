#!/usr/bin/env python
# encoding: utf-8

import unittest
import numpy as np

from ocupy import fixmat, utils

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.fm = fixmat.TestFixmatFactory(categories = [1,2,3], 
                                filenumbers = [1,2,3,4,5,6],
                                subjectindices = [1, 2, 3, 4, 5, 6],
                                params = {'pixels_per_degree':1, 'image_size':[100,500]})
        
    def test_imresize(self):
        arr = np.random.random((121, 111))
        arr_small = utils.imresize(arr, (121, 111))
        self.assertTrue(((arr-arr_small)**2).sum() < 10**-10)

    def test_ismember(self):
        a = np.array(range(1,100,2))
        for x in range(1,100,2):
            self.assertEquals(utils.ismember(x, a), [True])
        for x in range(2,100,2):
            self.assertEquals(utils.ismember(x, a), [False])
        base = range(1, 100, 2)
        base.extend(base)
        a = np.array(base)
        for x in range(1,100,2):
            self.assertEquals(utils.ismember(x, a.astype(float)), [True])
        for x in range(2,100,2):
            self.assertEquals(utils.ismember(x, a.astype(float)), [False])

    def test_dict_2_mat(self):
        d = {2:range(1,100), 3:range(1,10), 4: range(1,110)}
        self.assertRaises(RuntimeError, lambda: utils.dict_2_mat(d))
        
        d = {2:range(1,100), 3:range(1,100), 4: range(1,100)}
        m = utils.dict_2_mat(d)
        for c in range(1, 5):
            if c in d.keys():
                self.assertTrue(((np.array(d[c]) == m[c]).all()))
            else:
                self.assertTrue(np.isnan(m[c]).all())


if __name__ == '__main__':
    unittest.main()
