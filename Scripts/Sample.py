import pandas as pd
import numpy as np
import uuid
import os

import FormatDF as FDF
import Helpers as H


def get_sample(source_location, features, target, sample_size, custom_queries=[], threshold=0.5, even=True, feature_config='../../feature_config', save_seed=False, seed_directory='../../seeds'):
    df = pd.read_csv(source_location)
    df = FDF.format_column_names(df)
    for q in custom_queries:
        df = df.query(q)

    df = nan_feature_filter(df, features, feature_config)
    df = replace_null_features(df, feature_config)
    df = clean_outliers(df, target)
    if even:
        sample_df_all = random_sample(df, int(len(df) * 0.9))
        df_n = sample_df_all.query(str(target) + ' > ' + str(threshold))
        df_n = df_n.sample(frac=1)
        df_p = sample_df_all.query(str(target) + ' <= ' + str(threshold))
        df_p = df_p.sample(frac=1)
        sample_df = pd.concat([df_p.head(int(sample_size/2)), df_n.head(int(sample_size/2))], sort=False)
    else:
        sample_df = H.random_sample(df, sample_size)

    sample_df = sample_df.sample(frac=1)
    seed_folder = ''
    if save_seed and len(seed_directory) > 1:
        seed_id = str(uuid.uuid4().hex)
        seed_folder = seed_directory + '/' + seed_id
        os.mkdir(seed_folder)
        sample_df.to_csv(seed_folder + '/sample.csv', sep=',', encoding='utf-8')

    return sample_df, seed_folder


def clean_outliers(df, target):
    m = np.mean(df[target])
    s = np.std(df[target])
    df.query(target + ' > ' + str(m-2*s) + ' and ' + target + ' <= ' + str(m+2*s))
    return df


def nan_feature_filter(df, features, feature_config):
    nan_features = H.list_from_file(feature_config + '/NAN')
    acc = ''
    for i in range(len(nan_features)-1):
        if nan_features[i] in features:
            acc += str(nan_features[i]) + ' != "nan" and '

    acc += str(nan_features[len(nan_features)-1]) + ' != "nan"'
    df = df.query(acc)
    return df


def replace_null_features(df, feature_config):
    null_features = H.list_from_file(feature_config + '/NULL')
    for f in null_features:
        col_type = df.dtypes[df.columns.get_loc(f)]
        if col_type == "float64":
            df[f].fillna(0.0, inplace=True)
        else:
            df[f].fillna("0", inplace=True)
            df[f] = df[f].astype(str)
    return df


def random_sample(df, sample_size):
    sample_df = pd.DataFrame(data=None, columns=['sample_key'] + np.array(df.columns).tolist())
    sample = df.sample(n=sample_size, replace=False)
    sample_df[sample.columns] = sample[sample.columns]
    sample_df[['sample_key']] = np.array([i for i in range(len(sample_df))]).reshape(len(sample_df),1)
    return sample_df

