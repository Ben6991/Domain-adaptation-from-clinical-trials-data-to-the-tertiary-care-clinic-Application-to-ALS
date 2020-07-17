import sys

sys.path.append('C:/Users/Ben/PycharmProjects/master/')
import pandas as pd
import pickle
import TSFunctions as ts
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def save_pickle(x, file_name):
    print(f"saving file {file_name} ...", end=' ', flush=True)
    pickle.dump(x, open(file_name, "wb"), pickle.HIGHEST_PROTOCOL)
    print("saved", flush=True)


def split_patients(ids, test_frac, val_frac, early_frac, random_state):
    n_test_samples = int(test_frac * len(ids))
    n_val_samples = int(val_frac * len(ids))
    n_early_stopping_samples = int(early_frac * len(ids))

    train_pats, test_pats = train_test_split(ids, test_size=n_test_samples,
                                             random_state=random_state) if n_test_samples > 0 else [None, None]
    train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples,
                                            random_state=random_state) if n_val_samples > 0 else [None, None]
    train_pats, early_stop_pats = train_test_split(train_pats,
                                                   test_size=n_early_stopping_samples,
                                                   random_state=random_state
                                                   ) if n_early_stopping_samples > 0 else [None, None]

    return train_pats, val_pats, early_stop_pats, test_pats


def load_data(data_name):
    """ Load raw data, create y, drop patients without at least 3 visits, add 'no' drop patients without valid
    observations
    """
    if data_name == 'PROACT' or data_name == 'proact':
        data = None
        # data_file = f'C:/Users/Ben/Desktop/Study/Data/PROACT/data.csv'
        # data = pd.read_csv(data_file)
        # target = 'ALSFRS_Total'
        # data['y'] = data[target]
        # ts.drop_by_number_of_records(data, 3, inplace=True, verbose=2)
        # ts.add_ordering(data, inplace=True)
        pass
    else:
        # data_file = 'C:/Users/Ben/Desktop/Study/072019.csv'
        # data = pd.read_csv(data_file)
        # target = 'ALSFRS'
        # data['y'] = data[target]
        # ts.drop_by_number_of_records(data, 3, inplace=True, verbose=2)
        # ts.add_ordering(data, inplace=True)
        data = pd.read_csv('Exp1/f15_tasmc_processed.csv')

    return data


def path_to_proact_processed():
    return 'Exp2/f05_proact_processed.csv'


def preprocess_proact(data):
    to_drop = ['Next.FRS', 'Next.visit.Time', 'mouth', 'hands', 'trunk', 'leg', 'respiratory']
    data.drop(to_drop, 1, inplace=True)

    data.rename(columns={'Q1_Speech': 'Speech',
                         'Q2_Salivation': 'Salivation',
                         'Q3_Swallowing': 'Swallowing',
                         'Q4_Handwriting': 'Writing',
                         'Q5_Cutting': 'Food',
                         'Q6_Dressing_and_Hygiene': 'Dressing',
                         'Q7_Turning_in_Bed': 'Turning',
                         'Q8_Walking': 'Walking',
                         'Q9_Climbing_Stairs': 'Stairs',
                         'Q10_Respiratory': 'Dyspnea',
                         'Onset.time': 'TimeSinceOnset',
                         'ALSFRS_Total': 'ALSFRS',
                         'Sex.Male': 'Gender'},
                inplace=True)
    ts.add_missing_indicator(data, 'Gender', inplace=True)

    # save processed
    data['TimeSinceOnset'] *= (-1)
    data.to_csv(path_to_proact_processed(), index=False)


def get_features_to_scale():
    to_scale = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Turning',
                'Walking', 'Stairs', 'Dyspnea', 'ALSFRS', 'TimeSinceOnset', 'Age', 'time']
    return to_scale


def preprocess_tasmc(data):
    MMT = []

    FRS = ['Speech', 'Salivation', 'Swallowing',
           'Writing', 'Food', 'Dressing', 'Turning',
           'Walking', 'Stairs', 'Dyspnea']

    FRS_TOTAL = ['ALSFRS']

    # make TimeSinceInset constant and age
    if 'Age' not in data.columns:
        print("create age and time since onset")
        for ID in data.ID.unique():
            time_since_onset = data.loc[data['ID'] == ID, 'TimeSinceOnset'].min()
            data.loc[data['ID'] == ID, 'TimeSinceOnset'] = time_since_onset
            birth_year = data.loc[data['ID'] == ID, 'BirthYear'].values[0]
            year = int(data.loc[data['ID'] == ID, 'VisitDate'].values[0].split('-')[0])
            age_ID = year - birth_year
            data.loc[data['ID'] == ID, 'Age'] = age_ID
    #
    # # backward, forward and average interpolation
    # # ID: 763   Missing: 12,631     Rows without missing: 3,443  ID without missing: 626
    # data = ts.backward_imputation(data, FRS, only_if_max=4)
    # # ID: 763   Missing: 12,329     Rows without missing: 3,443  ID without missing: 626
    # if len(MMT) > 0:
    #     data = ts.backward_imputation(data, MMT, only_if_max=5)
    # # ID: 763   Missing: 11,887     Rows without missing: 3,457  ID without missing: 626
    # data = ts.backward_imputation(data, FRS_TOTAL, only_if_max=40)
    # # ID: 763   Missing: 11,886     Rows without missing: 3,457  ID without missing: 626
    # data = ts.forward_imputation(data, FRS + MMT + FRS_TOTAL, only_if_min=0)
    # # ID: 763   Missing: 11,010     Rows without missing: 3,464  ID without missing: 626
    # data = ts.simple_average_interpolation_imputation(data, FRS + MMT + FRS_TOTAL, only_if_equals=True)
    # # ID: 763   Missing: 7,782     Rows without missing: 3,510  ID without missing: 626
    # data['ALSFRS'].isna().sum()  # 190 missing in ALSFRS
    # data[FRS].sum(1, skipna=False).isna().sum()  # 161 missing in ALSFRS

    # data['ALSFRS'] = data[FRS].sum(1, skipna=False)

    if 'Education' in data.columns:
        data.drop(['Education', 'VisitDate', 'OnsetDate', 'BirthYear'], 1, inplace=True)

    data.loc[data['OnsetSite'] == 0, 'OnsetSite'] = -1
    data.loc[data['Gender'] == 0, 'Gender'] = -1

    ts.add_missing_indicator(data, 'OnsetSite', inplace=True)
    ts.add_missing_indicator(data, 'Gender', inplace=True)

    numeric = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Turning',
               'Walking', 'Stairs', 'Dyspnea', 'ALSFRS', 'TimeSinceOnset', 'Age']

    categorical = ['Gender', 'GenderNA', 'OnsetSite', 'OnsetSiteNA']

    # Complete missing with average
    # data = last_imputation(data, numeric=numeric, categorical=categorical)

    # save processed
    data.to_csv(path_to_tasmc_processed(), index=False)


def get_shared_features():
    temporal_features = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food',
                         'Dressing', 'Turning', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']
    constant_features = ['Gender', 'GenderNA', 'Onset.Limb', 'Onset.Bulbar', 'TimeSinceOnset', 'Age']
    return temporal_features, constant_features


def get_features_to_impute():
    numeric = ['RArmAbduct', 'RElbowFlex', 'RElbowExt', 'RFingerAbduct',
               'RFingerExt', 'LArmAbduct', 'LElbowFlex', 'LElbowExt', 'LFingerAbduct',
               'LFingerExt', 'RThighFlex', 'RKneeExt', 'RAnkleDorsi', 'RAnklePlantar',
               'LThighFlex', 'LKneeExt', 'LAnkleDorsi', 'LAnklePlantar', 'Speech',
               'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Turning',
               'Walking', 'Stairs', 'Dyspnea', 'ALSFRS', 'AgeOnset', 'TimeSinceOnset', 'Age']

    categorical = ['Gender', 'GenderNA', 'Sport', 'SportNA',
                   'OnsetSite', 'OnsetSiteNA', 'Familial', 'FamilialNA',
                   'Dementia', 'DementiaNA', 'isSmoking', 'isSmokingNA']

    return numeric, categorical


def amount_of_missing_values(data, exclude=None):
    return data.drop(exclude, 1).isna().sum().sum()


def path_to_tasmc_processed():
    return 'Exp2/f15_tasmc_processed.csv'


def data_preparation(data_name, run_process=False):
    """ 1. load data and pre-processing
        2. missing values rules (optional)
        3. scaling (optional)
    """
    data = load_data(data_name)

    if data_name == 'tasmc':
        data.loc[data['OnsetSite'] == -1, 'Onset.Bulbar'] = -1
        data.loc[data['OnsetSite'] == -1, 'Onset.Limb'] = 1

        data.loc[data['OnsetSite'] == 1, 'Onset.Bulbar'] = 1
        data.loc[data['OnsetSite'] == 1, 'Onset.Limb'] = -1

    if data_name == 'proact':
        #     if run_process:
        #         preprocess_proact(data)
        data = pd.read_csv(path_to_proact_processed())

    if data_name == 'tasmc':
        # if run_process:
        preprocess_tasmc(data)
        data = pd.read_csv(path_to_tasmc_processed())

    # define temporal and static features
    temporal_features, constant_features = get_shared_features()

    features = temporal_features + constant_features

    return data, features, temporal_features, constant_features


def drop_id_temp_1_if_exist(x):
    if 'ID_temp.1' in x.columns:
        x.drop(['ID_temp.1'], 1, inplace=True)


def drop_kurt_if_exist(x):
    kurt = [col for col in x.columns if col.split('_')[-1] == 'kurt']
    if len(kurt) > 0:
        x.drop(kurt, 1, inplace=True)


def get_proact_flat_data_path():
    return 'Exp2/f04_proact_flat.csv'


def proact_static():
    # import data (without scaling)
    data, features, temporal_features, constant_features = data_preparation('proact', run_process=True)

    # load mapping
    mapping = pickle.load(open("f00_proact_mapping.pkl", "rb"))

    # get flattened data
    flat_data = ts.flatten_data(data, mapping, temporal_features, constant_features,
                                ['kurt', 'skewness', 'aos', 'sos', 'soc', 'aoc'])
    flat_data.to_csv(get_proact_flat_data_path(), index=False)


def tasmc_static():
    # import data (without scaling)
    data, features, temporal_features, constant_features = data_preparation('tasmc', run_process=True)

    # # load mapping
    mapping_path = "f10_tasmc_mapping.pkl"
    mapping = pickle.load(open(mapping_path, 'rb'))

    # get flattened data
    print("flatten the data")
    flat_data = ts.flatten_data(data, mapping, temporal_features, constant_features,
                                exlude_features=['kurt', 'skewness', 'aos', 'sos', 'aoc', 'soc'])

    flat_data = downcast_x(flat_data)

    print("saving the flattened data...", end='', flush=True)
    flat_data.to_csv('Exp2/data/tasmc_static/tasmc_flat.csv', index=False)
    print("save")


def downcast_x(X):
    for column in X.columns:
        if X[column].dtype != 'O':
            X[column] = pd.to_numeric(X[column], downcast='integer')
            X[column] = pd.to_numeric(X[column], downcast='float')
            if X[column].min() >= 0:
                X[column] = pd.to_numeric(X[column], downcast='unsigned')
    return X


def data_setup(data, mapping_path, splits_info_path):
    # settings
    n_permutations = 60
    test_frac = 0.15
    val_frac = 0.15
    early_frac = 0.1

    # to use only once for each database
    create_and_save_mapping(data, mapping_path)

    # split
    create_and_save_splits_info(early_frac, test_frac, val_frac, n_permutations, splits_info_path, mapping_path)


def get_proact_temporal_data_path():
    return 'Exp2/data/proact_temporal/proact_temporal.pkl'


def proact_temporal():
    # import data
    data, features, temporal_features, constant_features = data_preparation('proact', run_process=False)

    # load mapping
    mapping = pickle.load(open('f00_proact_mapping.pkl', 'rb'))

    # scale and save proact_scaler
    print("scaling... ", end='', flush=True)
    to_scale = get_features_to_scale()
    scaler = MinMaxScaler((0, 0.5))
    data.loc[:, to_scale] = scaler.fit_transform(data.loc[:, to_scale])
    pickle.dump(scaler, open("f20_proact_scaler.pkl", "wb"))
    print("finished", flush=True)

    # create X and y for temporal models
    print("create and save temporal data", flush=True)
    create_and_save_temporal_data(data, features, mapping, get_proact_temporal_data_path())


def create_and_save_splits_info(early_frac, test_frac, val_frac, n_permutations, path, mapping_path):
    mapping = pickle.load(open(mapping_path, "rb"))
    splits_info = dict()
    for number_iter in range(n_permutations):
        train_pats, val_pats, early_stop_pats, test_pats = split_patients([*mapping.keys()],
                                                                          test_frac, val_frac, early_frac,
                                                                          number_iter)
        splits_info[number_iter] = dict(train_pats=train_pats,
                                        val_pats=val_pats,
                                        test_pats=test_pats,
                                        early_pats=early_stop_pats)
    pickle.dump(splits_info, open(path, "wb"))  # save


def create_and_save_temporal_data(data, features, mapping, path):
    print("create temporal lists ... ", end='', flush=True)
    X_list, y_list, lengths_list, idx_map = ts.create_temporal_Xy_lists(data, features, mapping,
                                                                        torch_tensors=True,
                                                                        verbose=1,
                                                                        t_pred=True,
                                                                        time_to_next=True,
                                                                        current_time=True,
                                                                        return_lengths=True,
                                                                        return_mapping_by_index=True)
    print("finished")
    print(f"saving {path}... ", end='', flush=True)
    pickle.dump([X_list, y_list, lengths_list, idx_map], open(path, "wb"))
    print("saved!")


def create_and_save_mapping(data, path):
    print("mapping enrichment...", end=" ", flush=True)
    mapping = ts.enrichment_by_ID_list(data, data['ID'].unique(), 2, progress_bar=True)
    print("finished.")
    print(f"save mapping as ({path})... ", end='', flush=True)
    pickle.dump(mapping, open(path, "wb"))
    print("saved!")


def min_max_scaling(data, features_to_scale, feature_range=(0.5, 1)):
    scaler = MinMaxScaler(feature_range)
    data.loc[:, features_to_scale] = scaler.fit_transform(data.loc[:, features_to_scale])
    return data


def frequent_imputation(data, features):
    imputer = SimpleImputer(strategy="most_frequent")
    data[features] = imputer.fit_transform(data[features])
    return data


def mean_imputation(data, features):
    imputer = SimpleImputer(strategy="mean")
    data[features] = imputer.fit_transform(data[features])
    if data[features].isna().sum().sum() > 0:
        raise Exception("Contain missing values")
    return data


def last_imputation(data, numeric=None, categorical=None):
    if numeric is not None:
        data = mean_imputation(data, numeric)
    if numeric is not None:
        data = frequent_imputation(data, categorical)
    return data


def tasmc_temporal():
    # import data
    data, features, temporal_features, constant_features = data_preparation('tasmc', run_process=False)

    # load mapping
    mapping = pickle.load(open('f10_tasmc_mapping.pkl', 'rb'))

    # create X and y for temporal models (for all observations)
    scaled_data = min_max_scaling(data.copy(), get_features_to_scale())
    create_and_save_temporal_data(scaled_data, features, mapping, f'Exp2/data/tasmc_temporal/tasmc_temporal.pkl')

    # scale and save proact_scaler
    print("scaling... ", end='', flush=True)
    to_scale = get_features_to_scale()
    scaler = pickle.load(open("f20_proact_scaler.pkl", "rb"))
    data.loc[:, to_scale] = scaler.transform(data.loc[:, to_scale])
    print("finished", flush=True)
    create_and_save_temporal_data(data, features, mapping,
                                  f'Exp2/data/tasmc_temporal/tasmc_temporal_scaled_by_proact.pkl')


def tasmc_setup():
    """
    create and save:    mapping (dict)
                        split_info (dict)
    """
    mapping_path = 'f10_tasmc_mapping.pkl'
    splits_info_path = 'f11_tasmc_split_info.pkl'
    data, features, temporal_features, constant_features = data_preparation('tasmc', run_process=True)
    data_setup(data, mapping_path, splits_info_path)


if __name__ == '__main__':
    # proact_static()  # PROACT flattened
    tasmc_static()  # TASMC flattened
    # proact_temporal()  # PROACT temporal
    tasmc_temporal()  # TASMC temporal
