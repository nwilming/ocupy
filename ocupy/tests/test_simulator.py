#!/usr/bin/env python
# encoding: utf-8

import pdb

import unittest
import numpy as np
from ocupy import fixmat
import simulator
import ocupy
from tool import KLDiv, multinomial, plot_probs
from IPython.kernel import client


class TestSimulator(unittest.TestCase):
    '''
    def test_init(self):
        # XXX Better use self generated data
        fm = fixmat.FixmatFactory('fixmat_photos.mat')
        gen = simulator.FixGen(fm)
        self.assertTrue(type(gen.fm)==ocupy.fixmat.FixMat)
        # In this data, the first fixation is deleted from the set
        self.assertTrue(gen.firstfixcentered == True)
        
        fm.fix-=1
        gen = simulator.FixGen(fm)
        self.assertTrue(gen.firstfixcentered == False)
        
    def test_anglendiff(self):
        fm = fixmat.FixmatFactory('/net/store/users/nwilming/eq/analysis/qoamp/fixmat_photos.mat')
        gen = simulator.FixGen(fm)
        
        for angle in [-180, -90, 0, 45, 90, 135]:
            print "Running anglendiff test on "+repr(angle)+" degrees."
            coord = [(0,0)]
            length = 1
            cur_angle = [np.nan]
            cur_angle.append(angle)
        
            # Create artificial fixmat with fixed angle differences
            for j in range(len(fm.x)-1):
                coord.append(simulator.calc_xy(coord[-1], cur_angle[-1], length))
                cur_angle.append(cur_angle[-1]+angle)
                
            fm.x = np.array([x[0] for x in coord])
            fm.y = np.array([x[1] for x in coord])
            
            #XXX Parameter should be None, not "None"
            gen.initialize_data(fit=None)
            
            # Use anglendiff to calculate angles and angle_diff
            a, l, ad, ld = simulator.anglendiff(fm, return_abs=True)
            a = np.round(a[0])  
            a[a==-180]=180
            
            cur_angle = simulator.reshift(np.array(cur_angle[0:-1]))
            
            # Assign nans to make arrays comparable
            cur_angle[np.isnan(a)]=np.nan
            cur_angle[np.round(cur_angle)==-180]=180
                
            ad = np.round(simulator.reshift(ad[0][~np.isnan(ad[0])]))
            
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
    '''
    def test_sim(self):
        ### XXX: Look out for the first angles and lengths!
        fm = fixmat.FixmatFactory('fixmat_photos.mat')
        sim = simulator.FixGen(fm)
        
        ### Test case 1: uniform dist
        H = np.ones((37,361))
        H/=H.sum()

        A = []
        for i in range(36):
            A.append(H)

        sim.initializeData(fit=None, full_H1=A)
        ### XXX: Evtl. numsamples anpassen an die menge der benutzten bins. 
        fixes = sim.sample_many(num_samples = 500)

        sim2 = simulator.FixGen(fixes)
        sim2.initializeData(fit=None)
        
        result_x = []
        result_y = []
        KLDiv_real_x = []
        KLDiv_real_y = []

        for i in range(len(sim.full_H1)-1):
            KLDiv_base_x = []
            KLDiv_base_y = []
            tmp = []
            tmp1 = []
            prob_base_x = []
            prob_base_y = []
            prob_real_x = []
            prob_real_y = []

            ### Draw 1000 samples from multinomial dist
            stat_base_x = np.random.multinomial(sim2.nosamples[i],np.sum(sim.full_H1[i],1),size=500)
            stat_base_y = np.random.multinomial(sim2.nosamples[i],np.sum(sim.full_H1[i],0),size=500)
            
            # Compute probabilities of stat_base
            for j in range(len(stat_base_x)):
                tmp.append(multinomial(stat_base_x[j], np.sum(sim.full_H1[i],1)))
                tmp1.append(multinomial(stat_base_y[j], np.sum(sim.full_H1[i],0)))
            prob_base_x.append(tmp)
            prob_base_y.append(tmp1)

            
            ### Compute KLDiv between 1000 samples and source
            for j in range(len(stat_base_x)):
                KLDiv_base_x.append(KLDiv(stat_base_x[j]/np.float(np.sum(stat_base_x[j])), np.sum(sim.full_H1[i],1)))
                KLDiv_base_y.append(KLDiv(stat_base_y[j]/np.float(np.sum(stat_base_y[j])), np.sum(sim.full_H1[i],0)))
                
            ### Compute KLDiv between simulated and source    
            KLDiv_real_x.append(KLDiv(np.sum(sim.full_H1[i],1), np.sum(sim2.full_H1[i],1)))
            KLDiv_real_y.append(KLDiv(np.sum(sim.full_H1[i],0), np.sum(sim2.full_H1[i],0)))
            
            # Compute probabilities of real outcomes
            prob_real_x.append(multinomial(np.sum(sim2.full_H1[i],1)*sim2.nosamples[i], np.sum(sim.full_H1[i],1)))
            prob_real_y.append(multinomial(np.sum(sim2.full_H1[i],0)*sim2.nosamples[i], np.sum(sim.full_H1[i],0)))            
            
            ### Compare simulated against multinomially sampled    
            result_x.append(np.array(KLDiv_real_x[-1]) > np.mean(KLDiv_base_x)+np.std(KLDiv_base_x))
            result_y.append(np.array(KLDiv_real_y[-1]) > np.mean(KLDiv_base_y)+np.std(KLDiv_base_y))
            pdb.set_trace()
        
        KLDiv_real_x, KLDiv_real_y = np.array(KLDiv_real_x), np.array(KLDiv_real_y)
            
        pdb.set_trace()

        if sum(result_x)<len(result_x):
            print "Horizontal margin failure in layer(s) " + repr(np.where(result_x))
            print "With a KLDiv of " + repr(KLDiv_real_x[result_x])
           
        if sum(result_y)<len(result_y):
            print "Vertical margin failure in layer(s) " + repr(np.where(result_y))
            print "With a KLDiv of " + repr(KLDiv_real_y[result_y])

def parallel_probs(sim, A):
    sim.initializeData(fit=None, full_H1=A)
    fixes = sim.sample_many(num_samples = 2000)
    sim2 = simulator.FixGen(fixes)
    sim2.initializeData(fit=None)
    
    stat_base = np.random.multinomial(sim2.nosamples[1],np.sum(sim.full_H1[1],1))
    
    
    return (multinomial(stat_base,np.sum(sim.full_H1[1],1)), multinomial(np.sum(sim2.full_H1[1],1)*sim2.nosamples[1], np.sum(sim.full_H1[1],1)))
    
    
def simprob():
    mec = client.MultiEngineClient()
    mec.get_ids()
    mec.execute('import simulator')
    mec.execute('from tool import multinomial')
    mec.execute('import numpy as np')
    fm = fixmat.FixmatFactory('fixmat_photos.mat')


    H = np.ones((37,361))
    H/=H.sum()
    A = []
    for i in range(36):
        A.append(H)

    sim = simulator.FixGen(fm)
    sim.initializeData(fit=None, full_H1 = A)   
    
    runs = 500
    siminst = [simulator.FixGen(fm) for x in range(runs)]
    A = [A for x in range(runs)]
    
    result = mec.map(parallel_probs, siminst, A)

    '''
    fix_list = mec.map(parallel_probs, siminst, A)
    sim2array = [simulator.FixGen(fixes) for fixes in fix_list]
    [sim2.initializeData(fit=None) for sim2 in sim2array]
    #pdb.set_trace()
   
    return [multinomial(np.sum(sim2.full_H1[1],1)*sim2.nosamples[1], np.sum(sim.full_H1[1],1)) for sim2 in sim2array]
    '''
    return result
        
        
if __name__ == '__main__':
    unittest.main()

