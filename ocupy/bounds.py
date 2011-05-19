#!/usr/bin/env python
"""This module implements functions for prediction bound computation."""

import numpy as np
from scipy.stats import nanmean

from ocupy import measures
from ocupy.fixmat import compute_fdm, NoFixations
from ocupy.utils import ismember


def intersubject_scores(fm, category, predicting_filenumbers,
                        predicting_subjects, predicted_filenumbers,
                        predicted_subjects, controls = True, scale_factor = 1):
    """
    Calculates how well the fixations from a set of subjects on a set of
    images can be predicted with the fixations from another set of subjects
    on another set of images.

    The prediction is carried out by computing a fixation density map from
    fixations of predicting_subjects subjects on predicting_images images.
    Prediction accuracy is assessed by measures.prediction_scores.

    Parameters
        fm : fixmat instance
        category : int
            Category from which the fixations are taken.
        predicting_filenumbers : list
            List of filenumbers used for prediction, i.e. images where fixations
            for the prediction are taken from.
        predicting_subjects : list
            List of subjects whose fixations on images in predicting_filenumbers
            are used for the prediction.
        predicted_filenumnbers : list
            List of images from which the to be predicted fixations are taken.
        predicted_subjects : list
            List of subjects used for evaluation, i.e subjects whose fixations
            on images in predicted_filenumbers are taken for evaluation.
        controls : bool, optional
            If True (default), n_predict subjects are chosen from the fixmat.
            If False, 1000 fixations are randomly generated and used for
            testing.
        scale_factor : int, optional
            specifies the scaling of the fdm. Default is 1.

    Returns
        auc : area under the roc curve for sets of actuals and controls
        true_pos_rate : ndarray
            Rate of true positives for every given threshold value.
            All values appearing in actuals are taken as thresholds. Uses lower
            sum interpolation.
        false_pos_rate : ndarray
            See true_pos_rate but for false positives.

    """
    predicting_fm = fm[
        (ismember(fm.SUBJECTINDEX, predicting_subjects)) &
        (ismember(fm.filenumber, predicting_filenumbers)) &
        (fm.category == category)]
    predicted_fm = fm[
        (ismember(fm.SUBJECTINDEX,predicted_subjects)) &
        (ismember(fm.filenumber,predicted_filenumbers))&
        (fm.category == category)]
    try:
        predicting_fdm = compute_fdm(predicting_fm, scale_factor = scale_factor)
    except NoFixations:
        predicting_fdm = None

    if controls == True:
        fm_controls = fm[
            (ismember(fm.SUBJECTINDEX, predicted_subjects)) &
            ((ismember(fm.filenumber, predicted_filenumbers)) != True) &
            (fm.category == category)]
        return measures.prediction_scores(predicting_fdm, predicted_fm,
            controls = (fm_controls.y, fm_controls.x))
    return measures.prediction_scores(predicting_fdm, predicted_fm, controls = None)

def intersubject_scores_random_subjects(fm, category, filenumber, n_train,
                                        n_predict, controls=True,
                                        scale_factor = 1):
    """
    Calculates how well the fixations of n random subjects on one image can
    be predicted with the fixations of m other random subjects.

    Notes
        Function that uses intersubject_auc for computing auc.

    Parameters
        fm : fixmat instance
        category : int
            Category from which the fixations are taken.
        filnumber : int
            Image from which fixations are taken.
        n_train : int
            The number of subjects which are used for prediction.
        n_predict : int
            The number of subjects to predict
        controls : bool, optional
            If True (default), n_predict subjects are chosen from the fixmat.
            If False, 1000 fixations are randomly generated and used for
            testing.
        scale_factor : int, optional
            specifies the scaling of the fdm. Default is 1.

    Returns
        tuple : prediction scores
    """
    subjects = np.unique(fm.SUBJECTINDEX)
    if len(subjects) < n_train + n_predict:
        raise ValueError("""Not enough subjects in fixmat""")
    # draw a random sample of subjects for testing and evaluation, according
    # to the specified set sizes (n_train, n_predict)
    np.random.shuffle(subjects)
    predicted_subjects  = subjects[0 : n_predict]
    predicting_subjects = subjects[n_predict : n_predict + n_train]
    assert len(predicting_subjects) == n_train
    assert len(predicted_subjects) == n_predict
    assert [x not in predicting_subjects for x in predicted_subjects]
    return intersubject_scores(fm, category, [filenumber], predicting_subjects,
        [filenumber], predicted_subjects,
        controls, scale_factor)

def upper_bound(fm, nr_subs = None, scale_factor = 1):
    """
    compute the inter-subject consistency upper bound for a fixmat.

    Input:
        fm : a fixmat instance
        nr_subs : the number of subjects used for the prediction. Defaults
                  to the total number of subjects in the fixmat minus 1
        scale_factor : the scale factor of the FDMs. Default is 1.
    Returns:
        A list of scores; the list contains one dictionary for each measure.
        Each dictionary contains one key for each category and corresponding
        values is an array with scores for each subject.
    """
    nr_subs_total = len(np.unique(fm.SUBJECTINDEX))
    if not nr_subs:
        nr_subs = nr_subs_total - 1
    assert (nr_subs < nr_subs_total)
    # initialize output structure; every measure gets one dict with
    # category numbers as keys and numpy-arrays as values
    intersub_scores = []
    for measure in range(len(measures.scores)):
        res_dict = {}
        result_vectors = [np.empty(nr_subs_total) + np.nan
                            for _ in np.unique(fm.category)]
        res_dict.update(zip(np.unique(fm.category), result_vectors))
        intersub_scores.append(res_dict)
    #compute inter-subject scores for every stimulus, with leave-one-out
    #over subjects
    for fm_cat in fm.by_field('category'):
        cat = fm_cat.category[0]
        for (sub_counter, sub) in enumerate(np.unique(fm_cat.SUBJECTINDEX)):
            image_scores = []
            for fm_single in fm_cat.by_field('filenumber'):
                predicting_subs = (np.setdiff1d(np.unique(
                    fm_single.SUBJECTINDEX),[sub]))
                np.random.shuffle(predicting_subs)
                predicting_subs = predicting_subs[0:nr_subs]
                predicting_fm = fm_single[
                    (ismember(fm_single.SUBJECTINDEX, predicting_subs))]
                predicted_fm = fm_single[fm_single.SUBJECTINDEX == sub]
                try:
                    predicting_fdm = compute_fdm(predicting_fm,
                        scale_factor = scale_factor)
                except NoFixations:
                    predicting_fdm = None
                image_scores.append(measures.prediction_scores(
                                        predicting_fdm, predicted_fm))
            for (measure, score) in enumerate(nanmean(image_scores, 0)):
                intersub_scores[measure][cat][sub_counter] = score
    return intersub_scores

def lower_bound(fm, nr_subs = None, nr_imgs = None, scale_factor = 1):
    """
    Compute the spatial bias lower bound for a fixmat.

    Input:
        fm : a fixmat instance
        nr_subs : the number of subjects used for the prediction. Defaults
                  to the total number of subjects in the fixmat minus 1
        nr_imgs : the number of images used for prediction. If given, the
                  same number will be used for every category. If not given,
                  leave-one-out will be used in all categories.
        scale_factor : the scale factor of the FDMs. Default is 1.
    Returns:
        A list of spatial bias scores; the list contains one dictionary for each
         measure. Each dictionary contains one key for each category and
        corresponding values is an array with scores for each subject.
    """
    nr_subs_total = len(np.unique(fm.SUBJECTINDEX))
    if nr_subs is None:
        nr_subs = nr_subs_total - 1
    assert (nr_subs < nr_subs_total)
    # initialize output structure; every measure gets one dict with
    # category numbers as keys and numpy-arrays as values
    sb_scores = []
    for measure in range(len(measures.scores)):
        res_dict = {}
        result_vectors = [np.empty(nr_subs_total) + np.nan
                            for _ in np.unique(fm.category)]
        res_dict.update(zip(np.unique(fm.category),result_vectors))
        sb_scores.append(res_dict)
    # compute mean spatial bias predictive power for all subjects in all
    # categories
    for fm_cat in fm.by_field('category'):
        cat = fm_cat.category[0]
        nr_imgs_cat = len(np.unique(fm_cat.filenumber))
        if not nr_imgs:
            nr_imgs_current = nr_imgs_cat - 1
        else:
            nr_imgs_current = nr_imgs
        assert(nr_imgs_current < nr_imgs_cat)
        for (sub_counter, sub) in enumerate(np.unique(fm.SUBJECTINDEX)):
            image_scores = []
            for fm_single in fm_cat.by_field('filenumber'):
                # Iterating by field filenumber makes filenumbers
                # in fm_single unique: Just take the first one to get the
                # filenumber for this fixmat
                fn = fm_single.filenumber[0]
                predicting_subs = (np.setdiff1d(np.unique(
                    fm_cat.SUBJECTINDEX), [sub]))
                np.random.shuffle(predicting_subs)
                predicting_subs = predicting_subs[0:nr_subs]
                predicting_fns = (np.setdiff1d(np.unique(
                    fm_cat.filenumber), [fn]))
                np.random.shuffle(predicting_fns)
                predicting_fns = predicting_fns[0:nr_imgs_current]
                predicting_fm = fm_cat[
                    (ismember(fm_cat.SUBJECTINDEX, predicting_subs)) &
                    (ismember(fm_cat.filenumber, predicting_fns))]
                predicted_fm = fm_single[fm_single.SUBJECTINDEX == sub]
                try:
                    predicting_fdm = compute_fdm(predicting_fm,
                        scale_factor = scale_factor)
                except NoFixations:
                    predicting_fdm = None
                image_scores.append(measures.prediction_scores(predicting_fdm,
                     predicted_fm))
            for (measure, score) in enumerate(nanmean(image_scores, 0)):
                sb_scores[measure][cat][sub_counter] = score
    return sb_scores
