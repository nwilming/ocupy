import numpy as np
from multiprocessing import pool
from scipy import nanmean
from ocupy.simulator import anglendiff, reshift
from scipy.optimize import leastsq

"""
These values are used by some of the scripts and determine bin width
for the raster plots.
"""
e_dist = np.arange(-10.5, 11.5, 1)
e_angle = np.arange(0, 181, 1)


def prepare_data(fm, max_back, dur_cap=700):
    '''
    Computes angle and length differences up to given order and deletes
    suspiciously long fixations.

    Input
        fm: Fixmat
            Fixmat for which to comput angle and length differences
        max_back: Int
            Computes delta angle and amplitude up to order max_back.
        dur_cap: Int
            Longest allowed fixation duration

    Output
        fm: Fixmat
            Filtered fixmat that aligns to the other outputs.
        durations: ndarray
            Duration for each fixation in fm
        forward_angle:
            Angle between previous and next saccade.

    '''
    durations = np.roll(fm.end - fm.start, 1).astype(float)
    angles, lengths, ads, lds = anglendiff(fm, roll=max_back, return_abs=True)
    # durations and ads are aligned in a way that an entry in ads
    # encodes the angle of the saccade away from a fixation in
    # durations
    forward_angle = abs(reshift(ads[0])).astype(float)
    ads = [abs(reshift(a)) for a in ads]
    # Now filter out weird fixation durations
    id_in = durations > dur_cap
    durations[id_in] = np.nan
    forward_angle[id_in] = np.nan
    return fm, durations, forward_angle, ads, lds


def saccadic_momentum_effect(durations, forward_angle,
                             summary_stat=nanmean):
    """
    Computes the mean fixation duration at forward angles.
    """
    durations_per_da = np.nan * np.ones((len(e_angle) - 1,))
    for i, (bo, b1) in enumerate(zip(e_angle[:-1], e_angle[1:])):
        idx = (
            bo <= forward_angle) & (
            forward_angle < b1) & (
            ~np.isnan(durations))
        durations_per_da[i] = summary_stat(durations[idx])
    return durations_per_da


def ior_effect(durations, angle_diffs, length_diffs,
               summary_stat=np.mean, parallel=True, min_samples=20):
    """
    Computes a measure of fixation durations at delta angle and delta
    length combinations.
    """
    raster = np.empty((len(e_dist) - 1, len(e_angle) - 1), dtype=object)
    for a, (a_low, a_upp) in enumerate(zip(e_angle[:-1], e_angle[1:])):
        for d, (d_low, d_upp) in enumerate(zip(e_dist[:-1], e_dist[1:])):
            idx = ((d_low <= length_diffs) & (length_diffs < d_upp) &
                   (a_low <= angle_diffs) & (angle_diffs < a_upp))
            if sum(idx) < min_samples:
                raster[d, a] = np.array([np.nan])
            else:
                raster[d, a] = durations[idx]
    if parallel:
        p = pool.Pool(3)
        result = p.map(summary_stat, list(raster.flatten()))
        p.terminate()
    else:
        result = list(map(summary_stat, list(raster.flatten())))
    for idx, value in enumerate(result):
        i, j = np.unravel_index(idx, raster.shape)
        raster[i, j] = value
    return raster


def predict_fixation_duration(
        durations, angles, length_diffs, dataset=None, params=None):
    """
    Fits a non-linear piecewise regression to fixtaion durations for a fixmat.

    Returns corrected fixation durations.
    """
    if dataset is None:
        dataset = np.ones(durations.shape)
    corrected_durations = np.nan * np.ones(durations.shape)
    for i, ds in enumerate(np.unique(dataset)):
        e = lambda v, x, y, z: (leastsq_dual_model(x, z, *v) - y)
        v0 = [120, 220.0, -.1, 0.5, .1, .1]
        id_ds = dataset == ds
        idnan = (
            ~np.isnan(angles)) & (
            ~np.isnan(durations)) & (
            ~np.isnan(length_diffs))
        v, s = leastsq(
            e, v0, args=(
                angles[
                    idnan & id_ds], durations[
                    idnan & id_ds], length_diffs[
                    idnan & id_ds]), maxfev=10000)
        corrected_durations[id_ds] = (durations[id_ds] -
                                      (leastsq_dual_model(angles[id_ds], length_diffs[id_ds], *v)))
        if params is not None:
            params['v' + str(i)] = v
            params['s' + str(i)] = s
    return corrected_durations


def subject_predictions(fm, field='SUBJECTINDEX',
                        method=predict_fixation_duration, data=None):
    '''
    Calculates the saccadic momentum effect for individual subjects.

    Removes any effect of amplitude differences.

    The parameters are fitted on unbinned data. The effects are
    computed on binned data. See e_dist and e_angle for the binning
    parameter.
    '''
    if data is None:
        fma, dura, faa, adsa, ldsa = prepare_data(fm, dur_cap=700, max_back=5)
        adsa = adsa[0]
        ldsa = ldsa[0]
    else:
        fma, dura, faa, adsa, ldsa = data
    fma = fma.copy()  # [ones(fm.x.shape)]
    sub_effects = []
    sub_predictions = []
    parameters = []
    for i, fmsub in enumerate(np.unique(fma.field(field))):
        id = fma.field(field) == fmsub
        #_, dur, fa, ads, lds = prepare_data(fmsub, dur_cap = 700, max_back=5)
        dur, fa, ads, lds = dura[id], faa[id], adsa[id], ldsa[id]
        params = {}
        _ = method(dur, fa, lds, params=params)
        ps = params['v0']
        ld_corrected = leastsq_only_dist(lds, ps[4], ps[5])
        prediction = leastsq_only_angle(fa, ps[0], ps[1], ps[2], ps[3])
        sub_predictions += [saccadic_momentum_effect(prediction, fa)]
        sub_effects += [saccadic_momentum_effect(dur - ld_corrected, fa)]
        parameters += [ps]
    return np.array(sub_effects), np.array(sub_predictions), parameters


def leastsq_dual_model(
        fa, dl, split, intercept, slope1, slope2, slope3, slope4):
    breakdummy = fa < split
    split2 = 0
    breakdummy2 = dl < split2
    reg_full = np.array([np.ones(fa.shape) * intercept,
                         slope1 * fa,
                         slope2 * ((fa - split) * breakdummy),
                         slope3 * dl,
                         slope4 * ((dl - split2) * breakdummy2)])
    return reg_full.sum(0)


def leastsq_only_dist(dl, slope3, slope4):
    split2 = 0
    breakdummy2 = dl < split2
    reg_full = np.array([slope3 * dl,
                         slope4 * ((dl - split2) * breakdummy2)])
    return reg_full.sum(0)


def leastsq_only_angle(fa, split, intercept, slope1, slope2):
    breakdummy = fa < split
    reg_full = np.array([np.ones(fa.shape) * intercept,
                         slope1 * fa,
                         slope2 * ((fa - split) * breakdummy)])
    return reg_full.sum(0)
