import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

'''
all functions assuming data format as follows:
                _ID_ | _order_ | _kind_ | _time_ | ... features ... | y
'''


def split_dict(dictionary, test_frac=0.2, random_state=32):
    '''
    Split dictionary to train-test (by key)
    '''
    # split the train_dict into train-validation (for early stopping)
    list_dict = list(dictionary.items()).copy()
    random.seed(random_state)
    random.shuffle(list_dict)
    train_size = int((1 - test_frac) * len(list_dict))
    train = dict(list_dict[:train_size])
    test = dict(list_dict[train_size:])
    return train, test


# ====================== Filtering ====================== #

def drop_by_number_of_records(data, minimum_records, id='ID', inplace=False, verbose=0):
    if verbose > 0:
        print('##############')
        print(f'Drop IDs with less than {minimum_records} records', flush=True)
    if not inplace:
        data = data.copy()
    n_IDs_before = data[id].nunique()
    n_samples_before = data.shape[0]
    n_visits = get_number_of_observations(data)
    indices_to_drop = data[data[id].isin(n_visits[n_visits.n_observations < minimum_records][id].unique())].index.values
    data.drop(indices_to_drop, inplace=True)
    n_IDs_after = data[id].nunique()
    n_samples_after = data.shape[0]
    if verbose > 1:
        print(f'Dropped {n_IDs_before - n_IDs_after} IDs out of {n_IDs_before}', flush=True)
        print(f'Dropped {n_samples_before - n_samples_after} rows out of {n_samples_before}', flush=True)
    if verbose > 0:
        print('##############')
    if not inplace:
        return data


def time_between_records(temporal_data, first, second, ID=None, ordered=True, order_column='no', time='time', id='ID'):
    if ordered:
        if ID is None:
            t1 = temporal_data.loc[temporal_data[order_column] == first, time].values[0]
            t2 = temporal_data.loc[temporal_data[order_column] == second, time].values[0]
        else:
            t1 = temporal_data.loc[(temporal_data[id] == ID) & (temporal_data[order_column] == first), time].values[0]
            t2 = temporal_data.loc[(temporal_data[id] == ID) & (temporal_data[order_column] == second), time].values[0]
        return t2 - t1
    else:
        print("TO DO")
        pass


# ================= Convect pandas to tensor or numpy ====================== #

def create_xy_by_no(temporal_data, ID, x_points, y_points, tensors=True, id='ID', order_col='no'):
    p_data = temporal_data[temporal_data[id] == ID].copy()
    p_data.loc[:, 'kind'] = 'z'
    p_data.loc[p_data[order_col].isin(x_points), 'kind'] = 'x'
    p_data.loc[p_data[order_col].isin(y_points), 'kind'] = 'y'
    x_i, y_i = create_xy(p_data, tensors=tensors)
    return x_i, y_i


def create_xy(temporal_block, tensors=True):
    data = temporal_block.copy()
    if not np.isin('TimeToNext', data.columns):
        data = data[data.kind.isin(['x', 'y'])]
        points = data.no.values
        data.loc[:, 'TimeToNext'] = -999
        for i in range(data.shape[0] - 1):
            time_to_next = time_between_records(data, points[i], points[i + 1])
            data.loc[data.no == points[i], 'TimeToNext'] = time_to_next
    x_i = data[data.kind == 'x'].drop(['ID', 'time', 'no', 'kind', 'y'], 1).values
    y_i = data.loc[data.kind == 'y', 'y'].values
    if tensors:
        x_i = torch.FloatTensor(x_i)
        y_i = torch.FloatTensor(y_i).view(1, -1)
    return x_i, y_i


# =====================================================

def get_slope_between_two_records(temporal_data, records, variable, ID, order_name='no', time='time', id='ID'):
    data = temporal_data.copy()
    no_1, no_2 = records
    val_1 = data.loc[(data[id] == ID) & (data[order_name] == no_1), variable].values[0]
    val_2 = data.loc[(data[id] == ID) & (data[order_name] == no_2), variable].values[0]
    time_1 = data.loc[(data[id] == ID) & (data[order_name] == no_1), time].values[0]
    time_2 = data.loc[(data[id] == ID) & (data[order_name] == no_2), time].values[0]
    slope = (val_2 - val_1) / (time_2 - time_1)
    return slope


def add_ordering(data, order_name='no', time='time', id='ID', inplace=True, verbose=0):
    if verbose == 1:
        print('##############')
        print(f'Adding \'{order_name}\' feature')
        print('##############')
    'add feature of order. for example, "No. 1, 2, 3, 4" for each id'
    if not inplace:
        data = data.copy()
    for ID in data[id].unique():
        id_data = data[data[id] == ID]
        n_observations = id_data.shape[0]
        indices = id_data.sort_values(time).index.values
        data.loc[indices, order_name] = np.arange(1, n_observations + 1)
    data[order_name] = data[order_name].astype(int)
    if not inplace:
        return data


def get_number_of_observations(temporal_data, id='ID', new_column_name='n_observations'):
    'return a DataFrame with: id, number of observations'
    data = temporal_data.copy()
    temp = data.groupby(id).size().reset_index()
    temp.columns = [id, new_column_name]
    return temp


def get_Xy_temporal(temporal_data, identifiers, temporal_predictors=None, id='ID', target='y', padding_value=-999):
    data = temporal_data.copy()
    X = []
    y = []
    lengths = []
    IDs = []
    max_length = 0
    for ID in identifiers:
        id_data = data[data[id] == ID]
        x_i = id_data.loc[:, temporal_predictors].values
        seq_length = x_i.shape[0]
        max_length = max(max_length, seq_length)
        y_all = id_data.loc[(id_data[id] == ID) & (id_data['kind'] == target), target].values
        for y_i in y_all:
            lengths.append(seq_length)
            X.append(x_i.reshape(seq_length, -1))
            y.append(y_i)
            IDs.append(ID)
    return X, y, lengths, IDs


def get_Xy_static(temporal_data, predictors, train_IDs,
                  temporal_numeric_features=None,
                  temporal_static_features=None,
                  primary_time_feature=None,
                  secondary_time_features=None,
                  id='ID', time='time', target='y',
                  return_info=False):
    info = dict()
    data = temporal_data.copy()
    X = pd.DataFrame()
    y = []
    for i, ID in enumerate(train_IDs):
        if i % 250 == 0:
            print(f'{i}/{len(train_IDs)}')
        id_x = data.loc[(data[id] == ID) & (data['kind'] == 'x'), :].sort_values(time)
        id_y = data[(data[id] == ID) & (data.kind == target)]
        x_indices = id_x.index.values
        feature_vector = []
        features_names = []
        feature_vector.append(ID)
        features_names.append(id)
        # secondary time features
        if secondary_time_features is not None:
            for feature in secondary_time_features:
                if np.isin(feature, predictors):
                    info[feature] = []
                    # mean
                    val = id_x[feature].mean()
                    name = feature + '_mean'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # std
                    val = id_x[feature].std()
                    if np.isnan(val):
                        val = 0
                    name = feature + '_sts'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
        if np.isin(primary_time_feature, predictors):
            info[primary_time_feature] = []
            # primary time feature
            feature = primary_time_feature
            # t_first
            val = id_x[feature].values[0]
            name = 't_first'
            feature_vector.append(val);
            features_names.append(name)
            info[feature].append(name)
            # t_first
            val = id_x[feature].values[-1]
            name = 't_last'
            feature_vector.append(val);
            features_names.append(name)
            info[feature].append(name)
            # t_range
            val = id_x[feature].values[-1] - id_x[feature].values[0]
            name = 't_range'
            feature_vector.append(val);
            features_names.append(name)
            info[feature].append(name)

        if temporal_numeric_features is not None:
            for feature in temporal_numeric_features:
                if np.isin(feature, predictors):
                    info[feature] = []
                    # max
                    val = id_x[feature].max()
                    name = feature + '_max'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # min
                    val = id_x[feature].min()
                    name = feature + '_min'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # first
                    val = id_x[feature].values[0]
                    name = feature + '_first'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # last
                    val = id_x[feature].values[-1]
                    name = feature + '_last'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # average
                    val = id_x[feature].mean()
                    name = feature + '_mean'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # std
                    val = id_x[feature].std()
                    if (np.isnan(val)):
                        val = 0
                    name = feature + '_std'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)

                    # if (primary_time_feature is not None) & (np.isin(primary_time_feature, predictors)):
                    #     val = (id_x[feature].values[-1] - id_x[feature].values[0]) / (id_x[primary_time_feature].values[-1] - id_x[primary_time_feature].values[0])
                    #     print(val)
                    #     name = feature + '_slope'
                    #     feature_vector.append(val); features_names.append(name)

        # t_pred
        t_last_minum_one = id_x.loc[x_indices[-1], time]
        t_last = id_y.loc[:, time].values[0]
        val = t_last - t_last_minum_one
        name = 't_pred'
        info['t_pred'] = [name]
        feature_vector.append(val);
        features_names.append(name)

        y.append(id_y['y'].values[0])
        X = X.append(dict(zip(features_names, feature_vector)),
                     ignore_index=True)
    if return_info:
        return X, y, info
    else:
        return X, y


def create_Xy_static_by_number_of_visits(temporal_data,
                                         dictionary,
                                         temporal_numeric_features=None,
                                         constant_features=None,
                                         id='ID',
                                         time='time', target='y',
                                         order_col='no',
                                         return_columns_names=False,
                                         progress_bar=True):
    'get dictionary with keys as Identifiers and the values are list of tuples [(x_p, y_p), ..., (x_p, y_p)]'
    'Ignoring missing values when calculate the features'
    info = dict()
    data = temporal_data.copy()
    X = pd.DataFrame()
    y = []

    if progress_bar:
        iterator = enumerate(tqdm([*dictionary.keys()]))
    else:
        iterator = enumerate([*dictionary.keys()])
    for i, ID in iterator:
        for x_p, y_p in dictionary[ID]:
            id_x = data.loc[(data[id] == ID) & (data[order_col].isin(x_p)), :].sort_values(time)
            id_y = data[(data[id] == ID) & (data[order_col].isin(y_p))]
            x_indices = id_x.index.values
            feature_vector = []
            features_names = []
            feature_vector.append(ID)
            features_names.append(id)

            if temporal_numeric_features is not None:
                for feature in temporal_numeric_features:
                    info[feature] = []
                    # max
                    val = id_x[feature].max()
                    name = feature + '_max'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # min
                    val = id_x[feature].min()
                    name = feature + '_min'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # first
                    val = id_x[feature].values[0]
                    name = feature + '_first'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # last
                    val = id_x[feature].values[-1]
                    name = feature + '_last'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # average
                    val = id_x[feature].mean()
                    name = feature + '_mean'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # std
                    val = id_x[feature].std()
                    if (np.isnan(val)):
                        val = 0
                    name = feature + '_std'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)
                    # slope (val_n - val_0)/(t_range)
                    t_range = id_x['time'].values[-1] - id_x['time'].values[0]
                    vals_no_NA = id_x[feature].dropna()
                    if vals_no_NA.size < 2:
                        val = np.nan
                    else:
                        val = (vals_no_NA.values[-1] - vals_no_NA.values[0]) / t_range
                    name = feature + '_slope'
                    feature_vector.append(val);
                    features_names.append(name)
                    info[feature].append(name)

            if constant_features is not None:
                for feature in constant_features:
                    val = id_x[feature].unique()[0]
                    name = feature
                    feature_vector.append(val);
                    features_names.append(name)

            # t_pred
            t_last_minus_one = id_x.loc[x_indices[-1], time]
            t_last = id_y.loc[:, time].values[0]
            val = t_last - t_last_minus_one
            name = 't_pred'
            info['t_pred'] = [name]
            feature_vector.append(val);
            features_names.append(name)

            y.append(id_y[target].values[0])
            X = X.append(dict(zip(features_names, feature_vector)),
                         ignore_index=True)
    if return_columns_names:
        return X, y, info
    else:
        return X, y


def create_temporal_Xy_lists(data,
                             features,
                             dictionary,
                             IDs=None,
                             torch_tensors=False,
                             verbose=0,
                             return_lengths=False,
                             id='ID',
                             order_by='no',
                             t_pred=False,
                             time_to_next=False,
                             current_time=False,
                             return_mapping_by_index=False):
    data = data.copy()
    features = features.copy()
    idx_map = dict()
    if t_pred:
        features += ['t_pred']
    if time_to_next:
        features += ['time_to_next']
    if current_time:
        features += ['time']

    # initialize empty lists
    X = []
    y = []
    lengths = [] if return_lengths else None

    for ID in tqdm(dictionary):
        idx_map[ID] = []
        for x_points, y_points in dictionary[ID]:
            idx_map[ID].append(len(X))
            if t_pred:
                t_pred_i = time_between_records(data, x_points[-1], y_points[-1], ID)
                data.loc[data[id] == ID, 't_pred'] = t_pred_i
            if time_to_next:
                time_to_next_ = data[(data[id] == ID) & (data[order_by].isin(x_points + y_points))][
                    'time'].diff().dropna().values
                data.loc[(data[id] == ID) & (data[order_by].isin(x_points)), 'time_to_next'] = time_to_next_

            x_i = data.loc[(data[id] == ID) & (data[order_by].isin(x_points)), features]
            y_i = data.loc[(data[id] == ID) & (data[order_by].isin(y_points)), 'y']

            if return_lengths:
                lengths.append(x_i.shape[0])

            if torch_tensors:
                x_i = torch.FloatTensor(x_i.values)
                y_i = torch.FloatTensor(y_i.values)

            X.append(x_i)
            y.append(y_i)

    if torch_tensors and return_lengths:
        lengths = torch.Tensor(lengths)

    if return_mapping_by_index:
        return X, y, lengths, idx_map
    return X, y, lengths


def scaling_wDictionary(data, test_dict, scaling, features_to_scale, feature_range=(0, 1)):
    """
    :param data: pd.DataFrame
    :param test_dict: dictionary
    the test dictionary, scaling based on all samples except y visits
    :param scaling: str ('stand' or 'MinMax')
    :param features_to_scale: list
    :param feature_range: tuple (from, to)
    relevant only if scaling is 'MinMax"
    :return:
    """
    df = data.copy()
    df = add_kind_by_dictionary(df, test_dict)
    scaler = None
    if scaling is not None:
        if scaling == 'stand':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scaling == 'MinMax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(df.loc[df['kind'] == 'x', features_to_scale])
    df.loc[:, features_to_scale] = scaler.transform(df.loc[:, features_to_scale])
    return df


# ========================= Missing data imputation ============================ #

def backward_imputation(temporal_data, features, list_IDs=None, id='ID', time='time', only_if_max=None):
    'case 1: NA-X-Y --> X-X-Y'
    m = len(features)
    data = temporal_data.copy()
    n_missing_at_start = data[features].isna().sum().sum()
    if list_IDs is None:
        list_IDs = data[id].unique()
    for i, feature in enumerate(features):
        for ID in list_IDs:  # Run over all the patients
            indices = data.loc[data[id] == ID].index  # get the indices
            values = data.loc[indices, feature].values  # get the values
            if np.isnan(data.loc[indices[0], feature]) & (not all(data.loc[indices, feature].isna())):
                index_of_first_not_null = np.argwhere(~np.isnan(values))[0][0]
                value = values[index_of_first_not_null]
                indices_to_impute = indices[:index_of_first_not_null]
                if only_if_max is not None:  # Kamin rule
                    if value == only_if_max:
                        data.loc[indices_to_impute, feature] = value
                else:
                    data.loc[indices_to_impute, feature] = value
    n_missing_at_end = data[features].isna().sum().sum()
    print(f'Completed {n_missing_at_start - n_missing_at_end} data points', flush=True)
    return data


def forward_imputation(temporal_data, features, list_IDs=None, id='ID', time='time', only_if_min=None):
    'case 1: X-Y-NA --> X-Y-Y'
    m = len(features)
    data = temporal_data.copy()
    n_missing_at_start = data[features].isna().sum().sum()
    if list_IDs is None:
        list_IDs = data[id].unique()
    for i, feature in enumerate(tqdm(features)):
        for ID in list_IDs:  # Run over all the patients
            indices = data.loc[data[id] == ID].index  # get the indices
            values = data.loc[indices, feature].values  # get the indices
            if np.isnan(data.loc[indices[-1], feature]) & (not all(data.loc[indices, feature].isna())):
                index_of_last_not_null = np.argwhere(~np.isnan(values))[-1][0]
                value = values[index_of_last_not_null]
                indices_to_impute = indices[index_of_last_not_null + 1:]
                if only_if_min is not None:
                    if value == only_if_min:
                        data.loc[indices_to_impute, feature] = value
                else:
                    data.loc[indices_to_impute, feature] = value
    n_missing_at_end = data[features].isna().sum().sum()
    print(f'Completed {n_missing_at_start - n_missing_at_end} data points', flush=True)
    return data


def simple_average_interpolation_imputation(temporal_data, features, list_IDs=None, id='ID', time='time',
                                            only_if_equals=False):
    m = len(features)
    data = temporal_data.copy()
    n_missing_at_start = data[features].isna().sum().sum()
    if list_IDs is None:
        list_IDs = temporal_data[id].unique()
    for i, feature in enumerate(tqdm(features)):
        for ID in list_IDs:
            values = data[data[id] == ID].sort_values(time)[feature].values
            if not all(np.isnan(values)):
                indices = data[data[id] == ID].sort_values(time)[feature].index.values
                indices_to_complete = []
                pre_value = np.nan
                for index in indices:
                    cur_value = data.loc[index, feature]
                    if not np.isnan(cur_value):
                        if only_if_equals:
                            cond = (cur_value == pre_value)
                        else:
                            cond = True
                        if (len(indices_to_complete) > 0) & (not np.isnan(pre_value)) & cond:
                            val = (cur_value + pre_value) / 2
                            data.loc[indices_to_complete, feature] = val
                            indices_to_complete = []
                        pre_value = cur_value
                    else:
                        if not np.isnan(pre_value):
                            indices_to_complete.append(index)
    n_missing_at_end = data[features].isna().sum().sum()
    print(f'Completed {n_missing_at_start - n_missing_at_end} data points', flush=True)
    return data


def add_kind_column_by_DataFrame(temporal_data, table, id='ID', order_column='no'):
    data = temporal_data.copy()
    data['kind'] = 'x'
    for i in range(table.shape[0]):
        ID = table.loc[i, id]
        y_visit = table.loc[i, order_column]
        data.loc[(data[id] == ID) & (~data[order_column].isin([y_visit])), 'kind'] = 'y'
    return data


def add_kind_by_dictionary(temporal_data, dictio, id='ID', order_column='no'):
    data = temporal_data.copy()
    data['kind'] = 'x'
    for ID in [*dictio.keys()]:
        for x_visits, y_visits in dictio[ID]:
            data.loc[(data['ID'] == ID) & (data['no'].isin(x_visits)), 'kind'] = 'x'
            data.loc[(data['ID'] == ID) & (data['no'].isin(y_visits)), 'kind'] = 'y'
    return data


def enrichment(data,
               constraint=dict(),
               n_random_samples=None,
               min_x_visits=2,
               n_patients='all'):
    if 'ID' not in data.columns:
        raise Exception('\'ID\' column is missing')
    if 'no' not in data.columns:
        raise Exception('\'no\' column is missing')

    # enriched_data = pd.DataFrame()
    n_IDs = data['ID'].nunique() if n_patients == 'all' else n_patients

    for i in tqdm(range(n_IDs)):
        ID = data['ID'].unique()[i]
        visits = data[data['ID'] == ID]['no'].values
        n_visits = visits.size
        counter = 1
        to_concat = []
        for visit in range(min_x_visits, n_visits):
            x_p = visits[:visit]
            for y_p in visits[visit:]:
                new_ID = str(ID) + '_' + str(counter)
                counter += 1

                ID_data = data[(data['ID'] == ID) & (data['no'].isin(list(x_p) + [y_p]))]
                ID_data.loc[:, 'ID'] = new_ID
                ID_data.loc[ID_data['no'].isin(list(x_p)), 'kind'] = 'x'
                ID_data.loc[ID_data['no'].isin([y_p]), 'kind'] = 'y'
                ID_data['src_ID'] = ID
                time_to_prev = ID_data['time'].diff().fillna(0).values
                time_to_next = ID_data['time'].shift(-1).fillna(0).values
                ID_data.loc[:, 'time_to_prev'] = time_to_prev
                ID_data.loc[:, 'time_to_next'] = time_to_next
                ID_data.loc[:, 't_pred'] = time_to_next[-1]
                to_concat.append(ID_data.copy())
                # enriched_data = pd.concat([enriched_data, ID_data]).reset_index(drop=True)
    enriched_data = pd.concat(to_concat)
    return enriched_data


def enrichment_by_ID_list(temporal_data, list_IDs, min_x_visits, target='y', progress_bar=False, constraint=dict(),
                          random_sample=None, with_replacement=False, random_state=32, window_size=24 * (365 / 12)):
    np.random.seed(random_state)
    dictionary = dict()
    iterator = tqdm(list_IDs) if progress_bar else list_IDs
    for ID in iterator:
        dictionary[ID] = []
        visits = temporal_data.loc[temporal_data['ID'] == ID, 'no'].values
        for i in range(min_x_visits, visits.size):
            x_visits = visits[:i]
            y_visits = visits[i:]
            for y_vis in y_visits:
                FLAG = False  # raise flag if this sample should enter the dictionary
                y_val = temporal_data.loc[(temporal_data['ID'] == ID) & (temporal_data['no'] == y_vis), target].values[
                    0]
                if ~np.isnan(y_val):
                    if 'max_prediction_time' in constraint:
                        t_ = time_between_records(temporal_data, x_visits[-1], y_vis, ID)
                        if t_ > constraint['max_prediction_time']:
                            FLAG = True
                    t0 = temporal_data.loc[(temporal_data['ID'] == ID) & (temporal_data['no'] == x_visits[0]), 'time'].item()
                    tn = temporal_data.loc[(temporal_data['ID'] == ID) & (temporal_data['no'] == y_vis), 'time'].item()
                    if (tn - t0) > window_size:
                        print("+")
                        FLAG = True
                    if x_visits.size >= min_x_visits:
                        samp = (x_visits.tolist(), [y_vis])
                        if not FLAG:
                            dictionary[ID].append(samp)
        if len(dictionary[ID]) == 0:
            del dictionary[ID]
        else:
            if random_sample is not None:
                import random
                if with_replacement:
                    dictionary[ID] = random.choices(dictionary[ID], k=random_sample)
                else:
                    dictionary[ID] = random.sample(dictionary[ID], k=min(len(dictionary[ID]), random_sample))
    return dictionary


# -------------------------- Plots ------------------------ #


def plot_patients(data, ID, feature, visits=None, pred=None, show=True, xlim=None, ylim=None, points=None, **kwargs):
    temp = data[(data['ID'] == ID) & (data['no'].isin(visits))]
    plt.scatter(temp['time'], temp[feature], **kwargs)
    plt.grid(which='both', alpha=0.4)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if show:
        plt.show()


def load_data(path_to_data, y=None, dropna=None):
    """
    :param path_to_data: str
    :param y: str
    feature name that will 'y'
    :param dropna: Boolean or list
    if True, pd.dropna(data) is execute
    if list, pd.dropna(list) is excecute
    :return: pandas.DataFrame
    data
    """
    import pandas as pd
    data = pd.read_csv(path_to_data)
    if dropna is not None:
        if dropna is True:
            data.dropna(inplace=True)
        if isinstance(dropna, list):
            data.dropna(subset=dropna, inplace=True)
    if y is not None:
        data['y'] = data[y]
    return data


def create_mapping(data, ID_list, min_x_visits=2, max_y_visits=4, random_state=32, method='r_random', target='y'):
    """
    :param data: pd.DataFrame
    :param ID_list: list
    :param min_x_visits: int
    :param max_y_visits: int
    :param random_state: int
    :param method: str
        options: 'random_r', 'n_samples'
    :param target: str
    :return: dictionary
        example:
        ID -> [(x_visits_1, y_visits_1), (x_visits_2, y_visits_2), ..., (x_visits_n, y_visits_n)]
    """
    np.random.seed(random_state)
    if ('no' not in data.columns) or ('ID' not in data.columns):
        raise Exception('\'no\' or \'ID\' doesnt exist in data')
    dictio = dict()
    for ID in ID_list:
        dictio[ID] = []
        ID_visits = data.loc[data['ID'] == ID, 'no'].values
        n_visits = ID_visits.size
        # ------- method 1: using random r last visits ------- #
        if method == 'r_random':
            try:
                r = np.random.randint(1, min(n_visits - min_x_visits, max_y_visits) + 1)
            except:
                print(n_visits, min_x_visits, max_y_visits, ID)
                raise Exception("Error")
            y_visits = ID_visits[-r:]
            x_visits = ID_visits[:-r]
            for y_vis in y_visits:
                # check validity
                if ~data.loc[(data['ID'] == ID) & (data['no'] == y_vis), target].isna().values[0]:
                    samp = (x_visits.tolist(), [y_vis])
                    dictio[ID].append(samp)
        # ---------------------------------------------------- #
        if method == 'n_samples':
            visits = data[data['ID'] == ID]['no'].values
            for xlast_idx in range(min_x_visits - 1, visits.size - 1):
                xlast = visits[xlast_idx]
                for y_idx in range(xlast_idx + 1, visits.size):
                    y_vis = visits[y_idx]
                    if ~data.loc[(data['ID'] == ID) & (data['no'] == y_vis), target].isna().values[0]:
                        samp = (np.arange(1, xlast + 1).tolist(), [y_vis])
                        dictio[ID].append(samp)
            if len(dictio[ID]) > 0:
                dictio[ID] = random.sample(dictio[ID], k=min(len(dictio[ID]), max_y_visits))

        # drop if there are no samples for the current ID
        if len(dictio[ID]) == 0:
            del dictio[ID]

    return dictio


def print_configuration(config):
    row_format = "{:>15}" * (len([*config.keys()]) + 1)
    print(row_format.format("", *[*config.keys()]))
    for i, row in enumerate(np.array([*config.values()]).reshape(-1, len([*config.keys()]))):
        print(row_format.format('', *row))


def add_missing_indicator(data, feature, suffix='NA', inplace=False, complete_missing_with=-1):
    if not inplace:
        data = data.copy()
    data.loc[:, feature + suffix] = -1
    data.loc[data[feature].isna(), feature + suffix] = 1
    data.loc[data[feature].isna(), feature] = complete_missing_with
    if not inplace:
        return data


def get_first(x):
    # get first valid
    return pd.Series([x.loc[x[col].first_valid_index(), col] for col in x.columns if col != 'ID'],
                     index=[col for col in x.columns if col != 'ID'])


def get_last(x):
    return pd.Series([x.loc[x[col].last_valid_index(), col] for col in x.columns if col != 'ID'],
                     index=[col for col in x.columns if col != 'ID'])


def average_of_changes(x):
    return x.diff().mean()


def sum_of_changes(x):
    return x.diff().sum()


def average_of_slopes(x):
    x = x.dropna()
    return (x.drop('ID', 1).diff() / x[['time']].diff().values).mean()


def sum_of_slopes(x):
    x = x.dropna()
    return (x.drop('ID', 1).diff() / x[['time']].diff().values).sum()


def std_time_between(x):
    return x.drop('ID', 1).diff().std(skipna=True)


def mean_time_between(x):
    return x.drop('ID', 1).diff().mean()


def temporal_to_static(data, dictionary, temporal_features=[], constant_features=[], type_of_ID=str,
                       exlude_features=[]):
    to_concatenate = []
    counter = 0
    for j, ID in enumerate(tqdm([*dictionary.keys()])):
        for i, (x_visits, y_visits) in enumerate(dictionary[ID]):
            # print(i, x_visits, y_visits)
            ID_i_data = data[(data['ID'] == ID) & (data['no'].isin(x_visits))].copy()
            ID_i_data['ID'] = str(ID) + '_' + str(i)  #
            t_pred = time_between_records(data, x_visits[-1], y_visits[-1], ID)
            ID_i_data['t_pred'] = t_pred
            all_missing = [col for col in ID_i_data.columns if all(ID_i_data[col].isna())]
            mean_ = data[all_missing].mean().values
            ID_i_data[all_missing] = mean_
            y_i = data.loc[(data['ID'] == ID) & (data['no'].isin(y_visits)), 'y'].values[-1]
            ID_i_data['y'] = y_i
            to_concatenate.append(ID_i_data)
            counter += 1

    temp_data = pd.concat(to_concatenate).reset_index(drop=True)
    # First
    first_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(get_first).add_suffix('_first').reset_index()
    new_data = first_.copy()

    # Last
    last_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(get_last).add_suffix('_last').reset_index()
    new_data = pd.merge(new_data, last_, on='ID', how='left')

    # Average of changes (aoc)
    if 'aoc' not in exlude_features:
        aoc_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(average_of_changes).add_suffix(
            '_aoc').reset_index()
        new_data = pd.merge(new_data, aoc_, on='ID', how='left')

    # Sum of changes
    if 'soc' not in exlude_features:
        soc_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(sum_of_changes).add_suffix(
            '_soc').reset_index()
        new_data = pd.merge(new_data, soc_, on='ID', how='left')

    # Average of slopes (aos)
    if 'aos' not in exlude_features:
        aos_ = temp_data[['ID'] + temporal_features + ['time']].groupby('ID').apply(average_of_slopes).drop('time',
                                                                                                            1).add_suffix(
            '_aos').reset_index()
        new_data = pd.merge(new_data, aos_, on='ID', how='left')

    if 'sos' not in exlude_features:
        # Sum of slopes (sos)
        sos_ = temp_data[['ID'] + temporal_features + ['time']].groupby('ID').apply(sum_of_slopes).drop('time',
                                                                                                        1).add_suffix(
            '_sos').reset_index()
        new_data = pd.merge(new_data, sos_, on='ID', how='left')

    if 'max' not in exlude_features:
        # Max
        max_ = temp_data[['ID'] + temporal_features].groupby('ID').max().add_suffix('_max').reset_index()
        new_data = pd.merge(new_data, max_, on='ID', how='left')

    if 'min' not in exlude_features:
        # Min
        min_ = temp_data[['ID'] + temporal_features].groupby('ID').min().add_suffix('_min').reset_index()
        new_data = pd.merge(new_data, min_, on='ID', how='left')

    if 'mean' not in exlude_features:
        # Mean
        mean_ = temp_data[['ID'] + temporal_features].groupby('ID').mean().add_suffix('_mean').reset_index()
        new_data = pd.merge(new_data, mean_, on='ID', how='left')

    if 'std' not in exlude_features:
        # Std
        std_ = temp_data[['ID'] + temporal_features].groupby('ID').std().add_suffix('_std').reset_index().fillna(0)
        new_data = pd.merge(new_data, std_, on='ID', how='left')

    if 'skewness' not in exlude_features:
        # Skewness
        skew_ = temp_data[['ID'] + temporal_features].groupby('ID').skew().add_suffix('_skew').reset_index()
        new_data = pd.merge(new_data, skew_, on='ID')

    if 'kurtosis' not in exlude_features:
        # Kurtosis
        kurtosis_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(pd.DataFrame.kurtosis).add_suffix(
            '_kurt').reset_index()
        new_data = pd.merge(new_data, kurtosis_, on='ID')

    # t_range
    t_range_ = temp_data[['ID', 'time']].groupby('ID').max().rename(columns=dict(time='t_range')).reset_index()
    new_data = pd.merge(new_data, t_range_, on='ID')

    # Slope
    slope_ = ((first_.set_index('ID') - last_.drop('ID', 1).values) / t_range_.drop('ID', 1).values)
    slope_.columns = [x.replace('_first', '_slope') for x in slope_.columns]
    slope_.reset_index(inplace=True)
    new_data = pd.merge(new_data, slope_, on='ID')

    # std of time between visits
    time_between_std_ = temp_data[['ID', 'time']].groupby('ID').apply(std_time_between).rename(
        columns=dict(time='time_between_std')).reset_index()
    time_between_std_.fillna(0, inplace=True)
    new_data = pd.merge(new_data, time_between_std_, on='ID')

    # Average of time between visits
    time_between_mean_ = temp_data[['ID', 'time']].groupby('ID').apply(mean_time_between).rename(
        columns=dict(time='time_between_mean')).reset_index()
    new_data = pd.merge(new_data, time_between_mean_, on='ID')

    # Number of visits
    n_visits_ = pd.DataFrame(temp_data[['ID', 'time']].groupby('ID').size().reset_index())
    n_visits_.columns = ['ID', 'n_visits']
    new_data = pd.merge(new_data, n_visits_, on='ID')

    # add t_pred
    temporal_flat = pd.merge(new_data, temp_data[['ID', 't_pred']].drop_duplicates(), on='ID')

    # add y
    temporal_flat = pd.merge(temporal_flat, temp_data[['ID', 'y']].drop_duplicates())

    temporal_flat['ID_temp'] = temporal_flat['ID'].apply(lambda x: x.split('_')[0]).astype(type_of_ID)

    static_features = data[['ID'] + constant_features].drop_duplicates().reset_index(drop=True)
    static_features['ID'] = static_features['ID'].astype(str)
    static_features.rename(columns={'ID': 'ID_temp'}, inplace=True)

    temporal_flat = pd.merge(temporal_flat, static_features, on='ID_temp')
    temporal_flat.drop('ID_temp', 1, inplace=True)

    all_features_columns = [x for x in [*temporal_flat.columns] if ((x != 'y') and (x != 'ID') and (x != 't_pred'))]
    temporal_flat = temporal_flat[['ID'] + all_features_columns + ['t_pred', 'y']]
    if temporal_flat.shape[0] != counter:
        raise Exception("NOT GOOD")
    return temporal_flat


def flatten_data(data, mapping, temporal_features=[], constant_features=[], exlude_features=[]):
    to_concatenate = []
    counter = 0
    for j, ID in enumerate(tqdm(mapping)):
        for i, (x_visits, y_visits) in enumerate(mapping[ID]):
            ID_i_data = data[(data['ID'] == ID) & (data['no'].isin(x_visits))].copy()
            ID_i_data['ID'] = str(ID) + '_' + str(i)  #
            t_pred = time_between_records(data, x_visits[-1], y_visits[-1], ID)
            ID_i_data['t_pred'] = t_pred
            all_missing = [col for col in temporal_features if all(ID_i_data[col].isna())]
            mean_ = data[all_missing].mean().values
            ID_i_data[all_missing] = mean_
            y_i = data.loc[(data['ID'] == ID) & (data['no'].isin(y_visits)), 'y'].values[-1]
            ID_i_data['y'] = y_i
            to_concatenate.append(ID_i_data)
            counter += 1

    temp_data = pd.concat(to_concatenate).reset_index(drop=True)

    # Create features
    # First
    first_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(get_first).add_suffix('_first').reset_index()
    new_data = first_.copy()

    # Last
    last_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(get_last).add_suffix('_last').reset_index()
    new_data = pd.merge(new_data, last_, on='ID', how='left')

    # Average of changes (aoc)
    if 'aoc' not in exlude_features:
        aoc_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(average_of_changes).add_suffix(
            '_aoc').reset_index()
        new_data = pd.merge(new_data, aoc_, on='ID', how='left')

    # Sum of changes
    if 'soc' not in exlude_features:
        soc_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(sum_of_changes).add_suffix(
            '_soc').reset_index()
        new_data = pd.merge(new_data, soc_, on='ID', how='left')

    # Average of slopes (aos)
    if 'aos' not in exlude_features:
        aos_ = temp_data[['ID'] + temporal_features + ['time']].groupby('ID').apply(average_of_slopes).drop('time',
                                                                                                            1).add_suffix(
            '_aos').reset_index()
        new_data = pd.merge(new_data, aos_, on='ID', how='left')

    if 'sos' not in exlude_features:
        # Sum of slopes (sos)
        sos_ = temp_data[['ID'] + temporal_features + ['time']].groupby('ID').apply(sum_of_slopes).drop('time',
                                                                                                        1).add_suffix(
            '_sos').reset_index()
        new_data = pd.merge(new_data, sos_, on='ID', how='left')

    if 'max' not in exlude_features:
        # Max
        max_ = temp_data[['ID'] + temporal_features].groupby('ID').max().add_suffix('_max').reset_index()
        new_data = pd.merge(new_data, max_, on='ID', how='left')

    if 'min' not in exlude_features:
        # Min
        min_ = temp_data[['ID'] + temporal_features].groupby('ID').min().add_suffix('_min').reset_index()
        new_data = pd.merge(new_data, min_, on='ID', how='left')

    if 'mean' not in exlude_features:
        # Mean
        mean_ = temp_data[['ID'] + temporal_features].groupby('ID').mean().add_suffix('_mean').reset_index()
        new_data = pd.merge(new_data, mean_, on='ID', how='left')

    if 'std' not in exlude_features:
        # Std
        std_ = temp_data[['ID'] + temporal_features].groupby('ID').std().add_suffix('_std').reset_index().fillna(0)
        new_data = pd.merge(new_data, std_, on='ID', how='left')

    if 'skewness' not in exlude_features:
        # Skewness
        skew_ = temp_data[['ID'] + temporal_features].groupby('ID').skew().add_suffix('_skew').reset_index()
        new_data = pd.merge(new_data, skew_, on='ID', how='left')

    if 'kurtosis' not in exlude_features and 'kurt' not in exlude_features:
        # Kurtosis
        kurtosis_ = temp_data[['ID'] + temporal_features].groupby('ID').apply(pd.DataFrame.kurtosis).add_suffix(
            '_kurt').reset_index()
        new_data = pd.merge(new_data, kurtosis_, on='ID')

    # t_range
    t_range_ = temp_data[['ID', 'time']].groupby('ID').max().rename(columns=dict(time='t_range')).reset_index()
    new_data = pd.merge(new_data, t_range_, on='ID')

    # Slope TODO: t_range suppose to be the time between the first and the last **valid** values for each feature
    slope_ = ((first_.set_index('ID') - last_.drop('ID', 1).values) / t_range_.drop('ID', 1).values)
    slope_.columns = [x.replace('_first', '_slope') for x in slope_.columns]
    slope_.reset_index(inplace=True)
    new_data = pd.merge(new_data, slope_, on='ID')

    # std of time between visits
    time_between_std_ = temp_data[['ID', 'time']].groupby('ID').apply(std_time_between).rename(
        columns=dict(time='time_between_std')).reset_index()
    time_between_std_.fillna(0, inplace=True)
    new_data = pd.merge(new_data, time_between_std_, on='ID')

    # Average of time between visits
    time_between_mean_ = temp_data[['ID', 'time']].groupby('ID').apply(mean_time_between).rename(
        columns=dict(time='time_between_mean')).reset_index()
    new_data = pd.merge(new_data, time_between_mean_, on='ID')

    # Number of visits
    n_visits_ = pd.DataFrame(temp_data[['ID', 'time']].groupby('ID').size().reset_index())
    n_visits_.columns = ['ID', 'n_visits']
    new_data = pd.merge(new_data, n_visits_, on='ID')

    # add t_pred
    temporal_flat = pd.merge(new_data, temp_data[['ID', 't_pred']].drop_duplicates(), on='ID')

    # add y
    temporal_flat = pd.merge(temporal_flat, temp_data[['ID', 'y']].drop_duplicates())

    temporal_flat['ID_temp'] = temporal_flat['ID'].apply(lambda x: x.split('_')[0]).astype(str)

    static_features = data[['ID'] + constant_features].drop_duplicates().reset_index(drop=True)
    static_features['ID'] = static_features['ID'].astype(str)
    static_features.rename(columns={'ID': 'ID_temp'}, inplace=True)

    temporal_flat = pd.merge(temporal_flat, static_features, on='ID_temp')
    # temporal_flat.drop('ID_temp', 1, inplace=True)

    all_features_columns = [x for x in [*temporal_flat.columns] if ((x != 'y') and (x != 'ID') and (x != 't_pred'))]
    temporal_flat = temporal_flat[['ID', 'ID_temp'] + all_features_columns + ['t_pred', 'y']]

    if temporal_flat.shape[0] != counter:
        raise Exception("NOT GOOD")

    # drop duplicates columns
    temporal_flat = temporal_flat.loc[:, ~temporal_flat.columns.duplicated()]
    return temporal_flat


def drop_rows_contain_missing(data, frac=0.8, verbose=1):
    bef = data.isna().sum().sum()
    data = data[data.isna().mean(1) < frac]
    aft = data.isna().sum().sum()
    if verbose > 0:
        print(f'>> Dropped {bef - aft} rows')
    return data


def add_missing_indicator(data, feature, suffix='NA', inplace=False, complete_missing_with=-1):
    if not inplace:
        data = data.copy()
    data.loc[:, feature + suffix] = -1
    data.loc[data[feature].isna(), feature + suffix] = 1
    data.loc[data[feature].isna(), feature] = complete_missing_with
    if not inplace:
        return data


def convert_y_to_sequence(dictionary):
    # example:
    # ([1, 2, 3], [4]) --> ([1, 2, 3], [2, 3, 4])
    dictionary = dictionary.copy()
    for ID in [*dictionary.keys()]:
        for i in range(len(dictionary[ID])):
            x_visits, y_visit = dictionary[ID][i]
            new_y_visits = x_visits[1:] + y_visit
            dictionary[ID][i] = (x_visits, new_y_visits)
    return dictionary


def split_x_p_y_p(d, ID, window_size, min_x_p=2, max_samples_for_ID=4):
    x_visits = []
    y_visits = []
    if d.shape[0] > min_x_p:
        last_t = d['time'].values[-1]
        n_visits = d.shape[0]
        thresh = last_t - window_size
        before_thresh = d.loc[d.time <= thresh, 'no'].values
        after_thresh = d.loc[d.time > thresh, 'no'].values
        if after_thresh.shape[0] == n_visits:
            # all the visits are in time time interval of y
            # [][y1, y2, ..., ym]
            x_visits = after_thresh[0:min_x_p].tolist()
            y_visits = after_thresh[min_x_p:]
            y_visits = np.random.choice(y_visits, min(y_visits.size, max_samples_for_ID), replace=False).tolist()

        if after_thresh.shape[0] > 2 and before_thresh.shape[0] > 1:
            # [x1, .., xn][y1, ..., ym]
            x_visits = np.concatenate([before_thresh, after_thresh[0:1]]).tolist()
            y_visits = after_thresh[1:].tolist()
            if len(y_visits) > 4:
                y_visits = np.random.choice(after_thresh[1:], max_samples_for_ID, replace=False)

        if after_thresh.shape[0] == 1:
            split_x_p_y_p(d.head(-1), ID, window_size)
        return x_visits, y_visits
    else:
        # if could find appropriate division
        print(f'Patient {ID} does not meet the requirements')
        return x_visits, y_visits


def create_dictionary_with_window_size(data, IDs, window_size, min_x_visits=2, max_samples=4, random_state=32):
    np.random.seed(random_state)
    counter_of_IDs_dropped = 0
    dictionary = dict()
    for pat in IDs:
        dictionary[pat] = []
        temp = data.loc[data['ID'] == pat, ['time', 'no']]
        x_p, y_p = split_x_p_y_p(temp, pat, window_size, min_x_visits, max_samples)
        if len(x_p) > 0 and len(y_p) > 0:
            for visit_number in y_p:
                y_true_i = data.loc[(data.ID == pat) & (data['no'] == visit_number), 'y'].values[0]
                if not np.isnan(y_true_i):
                    y_p = [visit_number]
                    dictionary[pat].append((x_p, y_p))
        if len(dictionary[pat]) == 0:
            counter_of_IDs_dropped += 1
            del dictionary[pat]
    return dictionary
