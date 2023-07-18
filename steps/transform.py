"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

def feature_engineering_pipeline(df: DataFrame):
    df = df.dropna(subset=['region', 'age', 'weight', 'height', 'howlong', 'gender', 'eat', 'train', 'background', 'experience', 'schedule', 'howlong', 'deadlift', 'candj', 'snatch', 'backsq', 'experience', 'background', 'schedule', 'howlong'])
    
    # Removing parameters not of interest + less popular events
    df = df.drop(columns=['affiliate', 'team', 'name', 'athlete_id', 'fran', 'helen', 'grace', 'filthy50', 'fgonebad', 'run400', 'run5k', 'pullups', 'train'])

    decline_dict = {'Decline to answer|': np.nan}
    df = df.replace(decline_dict)
    df = df.dropna(subset=['background', 'experience', 'schedule', 'howlong', 'eat'])

    # Encoding background data
    # Encoding background questions
    df['rec'] = np.where(df['background'].str.contains('I regularly play recreational sports'), 1, 0)
    df['high_school'] = np.where(df['background'].str.contains('I played youth or high school level sports'), 1, 0)
    df['college'] = np.where(df['background'].str.contains('I played college sports'), 1, 0)
    df['pro'] = np.where(df['background'].str.contains('I played professional sports'), 1, 0)
    df['no_background'] = np.where(df['background'].str.contains('I have no athletic background besides CrossFit'), 1, 0)

    # Delete nonsense answers
    df = df[~(((df['high_school'] == 1) | (df['college'] == 1) | (df['pro'] == 1) | (df['rec'] == 1)) & (df['no_background'] == 1))]

    # Encoding experience questions
    # Create encoded columns for experience response
    df['exp_coach'] = np.where(df['experience'].str.contains('I began CrossFit with a coach'), 1, 0)
    df['exp_alone'] = np.where(df['experience'].str.contains('I began CrossFit by trying it alone'), 1, 0)
    df['exp_courses'] = np.where(df['experience'].str.contains('I have attended one or more specialty courses'), 1, 0)
    df['life_changing'] = np.where(df['experience'].str.contains('I have had a life-changing experience due to CrossFit'), 1, 0)
    df['exp_trainer'] = np.where(df['experience'].str.contains('I train other people'), 1, 0)
    df['exp_level1'] = np.where(df['experience'].str.contains('I have completed the CrossFit Level 1 certificate course'), 1, 0)

    # Delete nonsense answers
    df = df[~((df['exp_coach'] == 1) & (df['exp_alone'] == 1))]


    # Creating no response option for coaching start
    df['exp_start_nr'] = np.where(((df['exp_coach'] == 0) & (df['exp_alone'] == 0)), 1, 0)

    # Other options are assumed to be 0 if not explicitly selected

    # Creating encoded columns with schedule data
    df['rest_plus'] = np.where(df['schedule'].str.contains('I typically rest 4 or more days per month'), 1, 0)
    df['rest_minus'] = np.where(df['schedule'].str.contains('I typically rest fewer than 4 days per month'), 1, 0)
    df['rest_sched'] = np.where(df['schedule'].str.contains('I strictly schedule my rest days'), 1, 0)

    df['sched_0extra'] = np.where(df['schedule'].str.contains('I usually only do 1 workout a day'), 1, 0)
    df['sched_1extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 1x a week'), 1, 0)
    df['sched_2extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 2x a week'), 1, 0)
    df['sched_3extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 3\+ times a week'), 1, 0)

    # Points are only assigned for the highest extra workout value (3x only vs. 3x and 2x and 1x if multi selected)
    df['sched_0extra'] = np.where((df['sched_3extra'] == 1), 0, df['sched_0extra'])
    df['sched_1extra'] = np.where((df['sched_3extra'] == 1), 0, df['sched_1extra'])
    df['sched_2extra'] = np.where((df['sched_3extra'] == 1), 0, df['sched_2extra'])
    df['sched_0extra'] = np.where((df['sched_2extra'] == 1), 0, df['sched_0extra'])
    df['sched_1extra'] = np.where((df['sched_2extra'] == 1), 0, df['sched_1extra'])
    df['sched_0extra'] = np.where((df['sched_1extra'] == 1), 0, df['sched_0extra'])

    # Adding no response columns
    df['sched_nr'] = np.where(((df['sched_0extra'] == 0) & (df['sched_1extra'] == 0) & (df['sched_2extra'] == 0) & (df['sched_3extra'] == 0)), 1, 0)
    df['rest_nr'] = np.where(((df['rest_plus'] == 0) & (df['rest_minus'] == 0)), 1, 0)

    # Scheduling rest days is assumed to be 0 if not explicitly selected

    # Encoding howlong (CrossFit lifetime)
    df['exp_1to2yrs'] = np.where((df['howlong'].str.contains('1-2 years')), 1, 0)
    df['exp_2to4yrs'] = np.where((df['howlong'].str.contains('2-4 years')), 1, 0)
    df['exp_4plus'] = np.where((df['howlong'].str.contains('4\+ years')), 1, 0)
    df['exp_6to12mo'] = np.where((df['howlong'].str.contains('6-12 months')), 1, 0)
    df['exp_lt6mo'] = np.where((df['howlong'].str.contains('Less than 6 months')), 1, 0)

    # Keeping only the highest response
    df['exp_lt6mo'] = np.where((df['exp_4plus'] == 1), 0, df['exp_lt6mo'])
    df['exp_6to12mo'] = np.where((df['exp_4plus'] == 1), 0, df['exp_6to12mo'])
    df['exp_1to2yrs'] = np.where((df['exp_4plus'] == 1), 0, df['exp_1to2yrs'])
    df['exp_2to4yrs'] = np.where((df['exp_4plus'] == 1), 0, df['exp_2to4yrs'])
    df['exp_lt6mo'] = np.where((df['exp_2to4yrs'] == 1), 0, df['exp_lt6mo'])
    df['exp_6to12mo'] = np.where((df['exp_2to4yrs'] == 1), 0, df['exp_6to12mo'])
    df['exp_1to2yrs'] = np.where((df['exp_2to4yrs'] == 1), 0, df['exp_1to2yrs'])
    df['exp_lt6mo'] = np.where((df['exp_1to2yrs'] == 1), 0, df['exp_lt6mo'])
    df['exp_6to12mo'] = np.where((df['exp_1to2yrs'] == 1), 0, df['exp_6to12mo'])
    df['exp_lt6mo'] = np.where((df['exp_6to12mo'] == 1), 0, df['exp_lt6mo'])

    # Encoding dietary preferences
    df['eat_conv'] = np.where((df['eat'].str.contains('I eat whatever is convenient')), 1, 0)
    df['eat_cheat'] = np.where((df['eat'].str.contains('I eat 1-3 full cheat meals per week')), 1, 0)
    df['eat_quality'] = np.where((df['eat'].str.contains("I eat quality foods but don't measure the amount")), 1, 0)
    df['eat_paleo'] = np.where((df['eat'].str.contains('I eat strict Paleo')), 1, 0)
    df['eat_cheat'] = np.where((df['eat'].str.contains('I eat 1-3 full cheat meals per week')), 1, 0)
    df['eat_weigh'] = np.where((df['eat'].str.contains('I weigh and measure my food')), 1, 0)

    # Encoding location as US vs non-US
    US_regions = ['Southern California', 'North East', 'North Central', 'South East', 'South Central', 'South West', 'Mid Atlantic', 'Northern California', 'Central East', 'North West']
    df['US'] = np.where((df['region'].isin(US_regions)), 1, 0)

    # Encoding gender
    df['gender_'] = np.where(df['gender'] == 'Male', 1, 0)

    df['BMI'] = df['weight'] * 0.453592 / np.square(df['height'] * 0.0254)

    df = df[(df['BMI'] >= 17) & (df['BMI'] <= 50)]  # Considers only underweight - morbidly obese competitors

    df['bmi_rounded'] = df['BMI'].round()

    df = df.drop(
        columns=['region', 'height', 'weight', 'candj', 'snatch', 'deadlift', 'norm_bs', 'norm_dl', 'norm_j', 'norm_s',
                 'bmi_rounded', 'backsq', 'eat', 'background', 'experience', 'schedule', 'howlong', 'gender'])

    return df

def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible transformer object.
    #
    # Identity feature transformation is applied when None is returned.

    import sklearn
    
    function_transformer_params = (
        {}
        if sklearn.__version__.startswith("1.0")
        else {"feature_names_out": "one-to-one"}
    )

    return Pipeline(
        steps = [
            (
                "feature_engineering",
                FunctionTransformer(feature_engineering_pipeline, **function_transformer_params),
            )
        ]
    )
