# -*- coding: utf-8 -*-
'''
Convert eye-tracking samples to fixations
'''
import numpy as np
from ocupy.datamat import AccumulatorFactory

velocity_window_size = 3


def get_velocity(samplemat, Hz, blinks=None):
    '''
    Compute velocity of eye-movements.

    Samplemat must contain fields 'x' and 'y', specifying the x,y coordinates
    of gaze location. The function assumes that the values in x,y are sampled
    continously at a rate specified by 'Hz'.
    '''
    Hz = float(Hz)
    distance = ((np.diff(samplemat.x) ** 2) +
                (np.diff(samplemat.y) ** 2)) ** .5
    distance = np.hstack(([distance[0]], distance))
    if blinks is not None:
        distance[blinks[1:]] = np.nan
    win = np.ones((velocity_window_size)) / float(velocity_window_size)
    velocity = np.convolve(distance, win, mode='same')
    velocity = velocity / (velocity_window_size / Hz)
    acceleration = np.diff(velocity) / (1. / Hz)
    acceleration = abs(np.hstack(([acceleration[0]], acceleration)))
    return velocity, acceleration


def saccade_detection(samplemat, Hz=200, threshold=30,
                      acc_thresh=2000, min_duration=21, min_movement=.35,
                      ignore_blinks=False):
    '''
    Detect saccades in a stream of gaze location samples.

    Coordinates in samplemat are assumed to be in degrees.

    Saccades are detect by a velocity/acceleration threshold approach.
    A saccade starts when a) the velocity is above threshold, b) the
    acceleration is above acc_thresh at least once during the interval
    defined by the velocity threshold, c) the saccade lasts at least min_duration
    ms and d) the distance between saccade start and enpoint is at least
    min_movement degrees.
    '''
    if ignore_blinks:
        velocity, acceleration = get_velocity(samplemat, float(Hz), blinks=samplemat.blinks)
    else:
        velocity, acceleration = get_velocity(samplemat, float(Hz))

    saccades = (velocity > threshold)
    #print velocity[samplemat.blinks[1:]]
    #print saccades[samplemat.blinks[1:]]

    borders = np.where(np.diff(saccades.astype(int)))[0] + 1
    if velocity[1] > threshold:
        borders = np.hstack(([0], borders))
    saccade = 0 * np.ones(samplemat.x.shape)

    # Only count saccades when acceleration also surpasses threshold
    for i, (start, end) in enumerate(zip(borders[0::2], borders[1::2])):
        if sum(acceleration[start:end] > acc_thresh) >= 1:
            saccade[start:end] = 1

    borders = np.where(np.diff(saccade.astype(int)))[0] + 1
    if saccade[0] == 0:
        borders = np.hstack(([0], borders))
    for i, (start, end) in enumerate(zip(borders[0::2], borders[1::2])):
        if (1000*(end - start) / float(Hz)) < (min_duration):
            saccade[start:end] = 1

    # Delete saccade between fixations that are too close together.
    dists_ok = False
    while not dists_ok:
        dists_ok = True
        num_merges = 0
        for i, (lfixstart, lfixend, start, end, nfixstart, nfixend) in enumerate(zip(
                borders[0::2], borders[1::2],
                borders[1::2], borders[2::2],
                borders[2::2], borders[3::2])):
            lastx = samplemat.x[lfixstart:lfixend].mean()
            lasty = samplemat.y[lfixstart:lfixend].mean()
            nextx = samplemat.x[nfixstart:nfixend].mean()
            nexty = samplemat.y[nfixstart:nfixend].mean()
            if (1000*(lfixend - lfixstart) / float(Hz)) < (min_duration):
                saccade[lfixstart:lfixend] = 1
                continue
            distance = ((nextx - lastx) ** 2 + (nexty - lasty) ** 2) ** .5
            if distance < min_movement:
                num_merges += 1
                dists_ok = False
                saccade[start:end] = 0
        borders = np.where(np.diff(saccade.astype(int)))[0] + 1
        if saccade[0] == 0:
            borders = np.hstack(([0], borders))
    return saccade.astype(bool)


def fixation_detection(samplemat, saccades, Hz=200, samples2fix=None,
                       respect_trial_borders=False, sample_times=None):
    '''
    Detect Fixation from saccades.

    Fixations are defined as intervals between saccades. This function
    also calcuates start and end times (in ms) for each fixation.
    Input:
        samplemat: datamat
            Contains the recorded samples and associated metadata.
        saccades: ndarray
            Logical vector that is True for samples that belong to a saccade.
        Hz: Float
            Number of samples per second.
        samples2fix: Dict
            There is usually metadata associated with the samples (e.g. the
            trial number). This dictionary can be used to specify how the
            metadata should be collapsed for one fixation. It contains
            field names from samplemat as keys and functions as values that
            return one value when they are called with all samples for one
            fixation. In addition the function can raise an 'InvalidFixation'
            exception to signal that the fixation should be discarded.
    '''
    if samples2fix is None:
        samples2fix = {}
    fixations = ~saccades
    acc = AccumulatorFactory()
    if not respect_trial_borders:
        borders = np.where(np.diff(fixations.astype(int)))[0] + 1
    else:
        borders = np.where(
            ~(np.diff(fixations.astype(int)) == 0) |
            ~(np.diff(samplemat.trial.astype(int)) == 0))[0] + 1

    fixations = 0 * saccades.copy()
    if not saccades[0]:
        borders = np.hstack(([0], borders))
    #lasts,laste = borders[0], borders[1]
    for i, (start, end) in enumerate(zip(borders[0::2], borders[1::2])):

        current = {}
        for k in samplemat.fieldnames():
            if k in list(samples2fix.keys()):
                current[k] = samples2fix[k](samplemat, k, start, end)
            else:
                current[k] = np.mean(samplemat.field(k)[start:end])
        current['start_sample'] = start
        current['end_sample'] = end
        fixations[start:end] = 1
        # Calculate start and end time in ms
        if sample_times is None:
            current['start'] = 1000 * start / Hz
            current['end'] = 1000 * end / Hz
        else:
            current['start'] = sample_times[start]
            current['end'] = sample_times[end]

        #lasts, laste = start,end
        acc.update(current)

    return acc.get_dm(params=samplemat.parameters()), fixations.astype(bool)


class InvalidFixation(Exception):
    pass
