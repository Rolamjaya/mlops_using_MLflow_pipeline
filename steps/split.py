"""
This module defines the following routines used by the 'split' step of the regression pipeline:

- ``create_dataset_filter``: Defines customizable logic for filtering the training, validation,
  and test datasets produced by the data splitting procedure. Note that arbitrary transformations
  should go into the transform step.
"""

from pandas import DataFrame, Series


def create_dataset_filter(dataset: DataFrame) -> Series(bool):
    """
    Mark rows of the split datasets to be additionally filtered. This function will be called on
    the training datasets.

    :param dataset: The {train,validation,test} dataset produced by the data splitting procedure.
    :return: A Series indicating whether each row should be filtered
    """
    # FIXME::OPTIONAL: implement post-split filtering on the dataframes, such as data cleaning.

    columns_to_check = ['region', 'age', 'weight', 'height', 'howlong', 'gender', 'eat', 'train', 'background', 'experience', 'schedule', 'howlong', 'deadlift', 'candj', 'snatch', 'backsq', 'experience', 'background', 'schedule', 'howlong']
    keep_rows = dataset[columns_to_check].isna().any(axis=1)

    import numpy as np
    # Removing declines to answer as the only response
    decline_dict = {'Decline to answer|': np.nan}
    dataset = dataset.replace(decline_dict)
    columns_to_check = ['background', 'experience', 'schedule', 'howlong', 'eat']
    keep_survey = dataset[columns_to_check].isna().any(axis=1)
    # Removing problematic entries
    keep_unproblematic = (dataset['weight'] < 1500) & (dataset['gender'] != '--') & (dataset['age'] >= 18) & (dataset['height'] < 96) & (dataset['height'] > 48)

    # Removing lifts above world record holding lifts
    keep_rational = (dataset['deadlift'] > 0) & (dataset['deadlift'] <= 1105) | ((dataset['gender'] == 'Female') & (dataset['deadlift'] <= 636))& (dataset['candj'] > 0) & (dataset['candj'] <= 395) & (dataset['snatch'] > 0) & (dataset['snatch'] <= 496) & (dataset['backsq'] > 0) & (dataset['backsq'] <= 1069)

    # Removing/correcting problematic responses
    dataset['rest_plus'] = np.where(dataset['schedule'].str.contains('I typically rest 4 or more days per month'), 1, 0)
    dataset['rest_minus'] = np.where(dataset['schedule'].str.contains('I typically rest fewer than 4 days per month'), 1, 0)
    dataset['rest_sched'] = np.where(dataset['schedule'].str.contains('I strictly schedule my rest days'), 1, 0)
    keep_response = ~((dataset['rest_plus'] == 1) & (dataset['rest_minus'] == 1))

    # Encoding background data
    # Encoding background questions
    dataset['rec'] = np.where(dataset['background'].str.contains('I regularly play recreational sports'), 1, 0)
    dataset['high_school'] = np.where(dataset['background'].str.contains('I played youth or high school level sports'), 1, 0)
    dataset['college'] = np.where(dataset['background'].str.contains('I played college sports'), 1, 0)
    dataset['pro'] = np.where(dataset['background'].str.contains('I played professional sports'), 1, 0)
    dataset['no_background'] = np.where(dataset['background'].str.contains('I have no athletic background besides CrossFit'), 1, 0)

    # Delete nonsense answers
    keep_sensible_1 = ~(((dataset['high_school'] == 1) | (dataset['college'] == 1) | (dataset['pro'] == 1) | (dataset['rec'] == 1)) & (dataset['no_background'] == 1))

    # Encoding experience questions
    # Create encoded columns for experience response
    dataset['exp_coach'] = np.where(dataset['experience'].str.contains('I began CrossFit with a coach'), 1, 0)
    dataset['exp_alone'] = np.where(dataset['experience'].str.contains('I began CrossFit by trying it alone'), 1, 0)
    dataset['exp_courses'] = np.where(dataset['experience'].str.contains('I have attended one or more specialty courses'), 1, 0)
    dataset['life_changing'] = np.where(dataset['experience'].str.contains('I have had a life-changing experience due to CrossFit'), 1, 0)
    dataset['exp_trainer'] = np.where(dataset['experience'].str.contains('I train other people'), 1, 0)
    dataset['exp_level1'] = np.where(dataset['experience'].str.contains('I have completed the CrossFit Level 1 certificate course'), 1, 0)

    # Delete nonsense answers
    keep_sensible_2 = ~((dataset['exp_coach'] == 1) & (dataset['exp_alone'] == 1))

    dataset['norm_dl'] = dataset['deadlift'] / dataset['weight']
    dataset['norm_j'] = dataset['candj'] / dataset['weight']
    dataset['norm_s'] = dataset['snatch'] / dataset['weight']
    dataset['norm_bs'] = dataset['backsq'] / dataset['weight']

    dataset['total_lift'] = dataset['norm_dl'] + dataset['norm_j'] + dataset['norm_s'] + dataset['norm_bs']

    dataset['BMI'] = dataset['weight'] * 0.453592 / np.square(dataset['height'] * 0.0254)

    keep_BMI = ((dataset['BMI'] >= 17) & (dataset['BMI'] <= 50))  # Considers only underweight - morbidly obese competitors

    print(dataset.shape)

    return ((~keep_rows)&(~keep_survey) & keep_unproblematic & keep_rational & keep_response & keep_sensible_1 & keep_sensible_2 & keep_BMI)
