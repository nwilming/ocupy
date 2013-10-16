from math import ceil
import numpy as np
import random
import spline_base

    
def anglendiff(fm, roll = 1, return_abs=False):
    angle_diffs = []
    length_diffs = []
    lengths = []
    angles  = []
    
    for r in range(1, roll+1):
        heights = (fm.y - np.roll(fm.y,r)).astype(float)
        widths = (fm.x - np.roll(fm.x,r)).astype(float)
        
        heights[fm.fix<=min(fm.fix)+r-1]=float('nan')
        widths[fm.fix<=min(fm.fix)+r-1]=float('nan')
        
        lengths.append((widths**2+heights**2)**.5)
        angles.append(np.degrees(np.arctan2(heights,widths)))
        
        length_diffs.append(lengths[0] - np.roll(lengths[r-1],1))
        
        # -360: straight saccades, -180: return saccades, 
                # 0: straight saccades, 180: return saccades, 
                # 360: no return saccades
        angle_diffs.append(angles[0] - np.roll(angles[r-1],1))
                
    if return_abs==True:
        return angles, lengths, angle_diffs, length_diffs
        
    else:
        return angle_diffs, length_diffs
            
def createHist(ld, ad, 
        bins=[np.linspace(-36.5,36.5,74), np.linspace(-0.5,180.5,182)]):
    H, xedges, yedges = np.histogram2d(ld[~np.isnan(ld)], ad[~np.isnan(ad)],
                bins=bins)
    H = H / sum(sum(H))
    H[:,0]*=2  
    H[:,-1]*=2
    return H
    
def compute_cumsum(fm, arg):
    if arg == 'la':
        ang, len, ad, ld = anglendiff(fm, return_abs=True)
        screen_diag = int(ceil((fm.image_size[0]**2+
                    fm.image_size[1]**2)**0.5))
        y_arg = len[0][np.roll(fm.fix==min(fm.fix),1)]
        x_arg = reshift(ang[0][np.roll(fm.fix==min(fm.fix),1)])
        bins = [range(screen_diag+1), np.linspace(-180.5,180.5,362)]
    
    elif arg == 'coo':
        indexes = fm.fix==min(fm.fix)
        y_arg = fm.y[indexes]
        x_arg = fm.x[indexes]
        bins = [range(fm.image_size[0]+1), range(fm.image_size[1]+1)]
    
    elif arg == 'len':
        trajLen = np.roll(fm.fix,1)[fm.fix==min(fm.fix)]
        val, borders = np.histogram(trajLen, bins=1000)
        cumsum = np.cumsum(val.astype(float) / val.sum())
        return cumsum, borders
    
    else:
        raise ValueError(("Not a valid argument, "+
            "choose from 'la', 'coo' and 'len'."))
        
    H = createHist(y_arg, x_arg, bins=bins)
    return np.cumsum(np.concatenate(H)), H.shape
    
def drawFrom(cumsum, borders=[]):
    if len(borders)==0:
        return (cumsum>=random.random()).nonzero()[0][0]
    else:
        return borders[(cumsum>=random.random()).nonzero()[0][0]]

def reshift(I):
    # Output -180 to +180
    if type(I)==list:
        I = np.array(I)
    
    if type(I)==np.ndarray:
        while((I>180).sum()>0 or (I<-180).sum()>0):
            I[I>180] = I[I>180]-360
            I[I<-180] = I[I<-180]+360

    if (type(I) == int or type(I)==np.float64 or 
        type(I)==float or type(I)==np.float):
        while (I>180 or I<-180):
            if I > 180:
                I-=360
            if I < -180:
                I+=360
    
    return I

def spline(ad,ld,collapse=True,xdim=[-36,36]):
    ld = ld[~np.isnan(ld)]
    ad = reshift(ad[~np.isnan(ad)]) 
    samples = zip(ld,ad)

    if collapse: # von 0 bis 181
        e_y = np.linspace(-36.5,36.5,74)
        e_x = np.linspace(-0.5,180.5,182)
        ad = abs(ad)
        K = createHist(ld,ad)
        H = spline_base.spline_pdf(np.array(samples), 
                e_y, e_x, nr_knots_y = 3, nr_knots_x = 19,hist=K)       

    else:
        e_x = np.linspace(-180.5,179.5,361)
        e_y = np.linspace(xdim[0],xdim[1],(xdim[1]*2)+1)
        ad[ad>179.5]-=360
        samples = zip(ld,ad)

        H = spline_base.spline_pdf(np.array(samples), e_y, e_x, 
                nr_knots_y = 3, nr_knots_x = 19)
    return H/H.sum()

