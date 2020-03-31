import numpy as np
import TSFunctions as ts
from sklearn.model_selection import ParameterSampler
import pickle
from keras.layers import LSTM, Dense, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time


def get_max_len(l):
    m = 0
    for x in l:
        m = max(m, x.shape[0])
    return m

class MyRNN():
    def __init__(self):
        print("Model: RNN", flush=True)
        self.best_params = None
        self.param_dist = None
        self.set_param_dist()

    def set_param_dist(self, param_dist=None):
        if param_dist is None:
            self.param_dist = dict(lr=[0.0001, 0.001, 0.01, 0.1, 1.0],
                                   hidden_size=np.random.randint(2 ** 3, 2 ** 11, 100), ## 2**2 ==> 2*11
                                   # dropout=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                   batch_size=2**np.arange(8, 12))
        else:
            self.param_dist = param_dist

    def randomSearch(self, data, train_dict, test_dict,
                     temporal_numeric_features, n_iter, verbose=0,
                     constant_features=[], param_dist=None, random_state=32,
                     max_epochs=np.inf, test_frac=None,
                     PATH=None, TUNING_PATH=None, only_one_iteration=None, verbose_Xy=0,
                     train_data=None, validation_data=None):

        # Create parameter distribution
        self.set_param_dist(param_dist) if param_dist is not None else None

        # define features
        features = temporal_numeric_features + constant_features

        # split train dictionary to train-validation (for early stopping)
        print(f'Split Train-Validation for early stopping...', flush=True)
        train_dict, val_dict = ts.split_dict(train_dict, test_frac, random_state)

        # create Train-Validation for early stopping
        if train_data is None:
            print(f'Create Xy train...', flush=True)
            X_train, y_train, train_len = self.get_Xy(data, train_dict, features, verbose_Xy)
        else:
            X_train, y_train = train_data
        if validation_data is None:
            print(f'Create Xy validation...', flush=True)
            X_val, y_val, val_len = self.get_Xy(data, test_dict, features, verbose_Xy)
        else:
            X_val, y_val = validation_data

        # pad sequences
        X_train_padded = pad_sequences(X_train, maxlen=get_max_len(X_train), dtype='float64', value=-999)
        X_val_padded = pad_sequences(X_val, maxlen=get_max_len(X_val), dtype='float64', value=-999)

        # create parameter sampler for RandomSearch
        param_sampler = ParameterSampler(self.param_dist, n_iter=n_iter, random_state=random_state)

        # RandomSearch
        print(">> Random Search", flush=True)


        # split train-validation for early stopping
        X_train_train, X_early, y_train_train, y_early = train_test_split(X_train_padded, y_train,
                                                                          test_size=test_frac,
                                                                          random_state=random_state)

        min_rmse = np.inf
        best_params = None
        tuning_dict = dict()
        for i, param in enumerate(param_sampler):
            print(f'Param number {i}', flush=True)
            print(f'>> -------- Iteration {i + 1}/{n_iter} -------- <<', flush=True)
            for a, b in list(param.items()):
                print(f'{a}: {b}', flush=True)

            # Fit model
            early_stopping = EarlyStopping(patience=20)
            model_checkpoint = ModelCheckpoint(PATH, save_best_only=True, save_weights_only=True)
            model = self.create_model(hidden_size=param['hidden_size'], input_size=len(features) + 1, lr=param['lr'])
            history = model.fit(x=X_train_train,
                                y=np.array(y_train_train),
                                batch_size=min(param['batch_size'], X_train_train.shape[0]),
                                epochs=max_epochs,
                                callbacks=[early_stopping, model_checkpoint],
                                validation_data=(X_early, np.array(y_early)),
                                verbose=verbose)

            # validation evaluation
            print('validation evaluation', flush=True)
            new_model = self.create_model(hidden_size=param['hidden_size'], input_size=len(features) + 1, lr=param['lr'])
            new_model.load_weights(PATH)
            val_preds = model.predict(X_val_padded)
            rmse = mean_squared_error(np.array(y_val), val_preds) ** 0.5

            # save parameters if are the best
            print(f'improved?', flush=True)
            if rmse < min_rmse:
                print(f'Yes!')
                min_rmse = rmse
                best_params = param.copy()
            else:
                print(f'No!')
            # insert validation score to the dictionary
            tuning_dict[i] = (param, rmse)

        self.best_params = best_params
        if TUNING_PATH is not None:
            try:
                print("Saving tuning information", flush=True)
                with open(TUNING_PATH, 'wb') as f:
                    pickle.dump(tuning_dict, f, pickle.HIGHEST_PROTOCOL)
            except:
                print("Could not save tuning information", flush=True)

    def test_evaluation(self,
                        data,
                        train_dict,
                        test_dict,
                        temporal_numeric_features,
                        constant_features=[],
                        verbose=0,
                        early_stopping_frac=0.2,
                        max_epochs=1000,
                        PATH=None,
                        random_state=32,
                        train_data=None,
                        test_data=None,
                        verboseXy=0):

        features = temporal_numeric_features + constant_features
        # create Train-Validation for early stopping
        if train_data is None:
            print(f'Create Xy train...', flush=True)
            X_train, y_train, train_len = self.get_Xy(data, train_dict, features, verboseXy)
        else:
            X_train, y_train = train_data
        if test_data is None:
            print(f'Create Xy validation...', flush=True)
            X_test, y_test, test_len = self.get_Xy(data, test_dict, features, verboseXy)
        else:
            X_test, y_test = test_data

        # pad sequences
        X_train_padded = pad_sequences(X_train, maxlen=get_max_len(X_train), dtype='float64', value=-999)
        X_val_padded = pad_sequences(X_test, maxlen=get_max_len(X_test), dtype='float64', value=-999)

        param = self.best_params
        # split train-validation for early stopping
        X_train_train, X_early, y_train_train, y_early = train_test_split(X_train_padded, y_train,
                                                                          test_size=early_stopping_frac,
                                                                          random_state=random_state)

        # Fit model
        early_stopping = EarlyStopping(patience=20)
        model_checkpoint = ModelCheckpoint(PATH, save_best_only=True, save_weights_only=True)
        model = self.create_model(hidden_size=param['hidden_size'], input_size=len(features) + 1, lr=param['lr'])
        history = model.fit(x=X_train_train,
                            y=np.array(y_train_train),
                            batch_size=min(param['batch_size'], X_train_train.shape[0]),
                            epochs=max_epochs,
                            callbacks=[early_stopping, model_checkpoint],
                            validation_data=(X_early, np.array(y_early)))

        # validation evaluation
        model = self.create_model(hidden_size=param['hidden_size'], input_size=len(features) + 1, lr=param['lr'])

        print('load weights')
        model.load_weights(PATH)
        test_preds = model.predict(X_val_padded)

        return y_test, test_preds

    def create_model(self, hidden_size, lr, input_size):
        model = Sequential()
        model.add(Masking(mask_value=-999, input_shape=(None, input_size)))
        model.add(LSTM(units=hidden_size, input_shape=(None, input_size)))
        model.add(Dense(1, activation='linear'))
        adam = Adam(lr=lr)
        model.compile(optimizer=adam, loss='mse')
        return model

    def get_Xy(self, data, dictionary, features, verbose=0):
        from tqdm import tqdm
        X = []
        y = []
        lengths = []
        iterator = [*dictionary.keys()] if verbose == 0 else tqdm([*dictionary.keys()])
        for ID in iterator:
            for x_visits, y_visits in dictionary[ID]:
                x_i = data.loc[(data['ID'] == ID) & (data['no'].isin(x_visits)), features].values
                y_i = data.loc[(data['ID'] == ID) & (data['no'].isin(y_visits)), 'y'].values
                a = x_i.copy()
                t_i = ts.time_between_records(data, x_visits[-1], y_visits[0], ID)
                b = np.repeat(t_i, x_i.shape[0]).reshape(-1, 1)
                x_i = np.concatenate((a, b), 1)
                X.append(x_i)
                y.append(y_i)
                lengths.append(x_i.shape[0])
        return X, y, lengths


