import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import Helpers as H


def encode_inputs(df, features, target, feature_config='../../feature_config', one_hot=True):
    df = df[features + [target]]
    excep_features = H.list_from_file(feature_config + '/EXCEP')
    exclude_features = H.list_from_file(feature_config + '/EXCLUDE')

    df_encoded, ind_map = encode_category_features(df, excep_features, exclude_features, one_hot=one_hot)
    used_features = [x for x in features if x not in exclude_features]
    df_used = df_encoded[used_features + [target]]
    df_format = format_for_flat_input(df_used)
    df_labels = []
    for i in range(len(df_format)):
        df_labels.append(df_format[i].pop(-1))
    return df_format, df_labels, ind_map


def format_column_names(df):
    cols = df.columns
    cols = cols.map(lambda x: x.replace(' ', '_'))
    df.columns = cols
    return df


def encode_category_features(df, excep_features=[], exclude_features=[], one_hot=True):
    ind = 0
    ind_map = {}
    for col in df.columns:
        ind_range = []
        ind_range.append(ind)
        col_type = df.dtypes[df.columns.get_loc(col)]
        if col not in exclude_features:
            if (col in excep_features) or (col_type == 'O'):
                values = np.array(df[[col]])
                rvalues = values.ravel()
                if one_hot:
                    integer_encoded = rvalues.reshape(len(rvalues), 1)
                    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
                    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                    ind += len(onehot_encoded[0])
                    df[col] = onehot_encoded.tolist()
                else:
                    label_encoder = LabelEncoder()
                    integer_encoded = label_encoder.fit_transform(values)
                    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                    ind += len(integer_encoded[0])
                    df[col] = integer_encoded.tolist()
            else:
                ind += 1
            ind_range.append(ind)
            ind_map[col] = ind_range
    return df, ind_map


def date_diff(df, d1, d2, t):
    a1 = pd.to_datetime(df[d1])
    b1 = pd.to_datetime(df[d2])
    df[t] = ((a1-b1) / np.timedelta64(1, 'D')).astype(int)
    return df


def format_for_flat_input(df):
    list_type = type([])
    step1 = df.values.tolist()
    step2 = [[y if type(y) == list_type else [y] for y in x] for x in step1]
    step3 = [[z for y in x for z in y] for x in step2]
    return step3

