import unittest

import numpy as np

from ocupy import fixmat, stimuli, loader
from ocupy.xvalidation import SimpleXValidation

class TestXValidation(unittest.TestCase):
 
    def test_xvalidation(self):       
        img_per_cat = {}
        [img_per_cat.update({cat: range(1,50)}) for cat in range(1,11)]
        fm = fixmat.TestFixmatFactory(categories = range(1,11), subjectindices = range(1,11),
                                      filenumbers = range(1,11))
        l = loader.TestLoader(img_per_cat,size = (10,10))
        stim = stimuli.FixmatStimuliFactory(fm,l) 
        img_ratio = 0.3
        sub_ratio = 0.3
        data_slices = SimpleXValidation(fm, stim,img_ratio,sub_ratio,10)
        for i,(fm_train, cat_train, fm_test, cat_test) in enumerate(data_slices.generate()):
            self.assertEquals (len(np.unique(fm_train.SUBJECTINDEX)), 
               round(len(np.unique(fm.SUBJECTINDEX))*(1-sub_ratio))) 
            self.assertEquals(len(np.unique(fm_test.SUBJECTINDEX)), 
                   round(len(np.unique(fm.SUBJECTINDEX))*(sub_ratio))) 
            for (test,train,reference) in zip(cat_test,
                                               cat_train, 
                                               stim):
                self.assertEquals(test.category, reference.category)
                self.assertEquals(len(test.images()), 
                        len(reference.images())*img_ratio)
                self.assertEquals(len(train.images()),
                        len(reference.images())*(1-img_ratio))

                self.assertTrue((np.sort(np.unique(fm_train[fm_train.category==test.category].filenumber)) == 
                       np.sort( train.images())).all())

if __name__ == '__main__':
    unittest.main()
