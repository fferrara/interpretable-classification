from copy import deepcopy

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import xgboost
from sklearn.utils import shuffle
from pdpbox.pdp import pdp_isolate, pdp_plot
from eli5.sklearn import PermutationImportance
import warnings


class TitanicSurvival:
    def __init__(self, path):
        df = pd.read_csv(path)
        df_test = pd.read_csv('/home/flavio/data/Titanic/test_labels.csv')

        self.y = df.Survived
        self.X = df.drop('Survived', axis=1)
        self.X_train = None
        self.fitted_model = None

        self.test = df_test.drop(['Survived'], axis=1)
        self.y_test = df_test.Survived
        self.X_test = None

        self.search_iterations = 20
        self.search_space = [
            Integer(20, 81, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Integer(1, 12, name='min_child_weight'),

            Real(0, 0.3, "uniform", name='gamma'),
            Real(0.5, 0.9, "uniform", name='subsample'),
            Real(0.5, 0.9, "uniform", name='colsample_bytree')
        ]


    def _repr_html_(self):
        print('---------------------- Features -----------------------')
        print(self.X.info())
        print()
        print('---------------------- Target distribution -----------------------')
        print(self.y.hist())

    def preprocess(self):
        X_train = self.X.copy()
        X_test = self.test.copy()

        columns_missing_values = self.X.columns[self.X.isnull().any()]
        for column in columns_missing_values:
            if is_numeric_dtype(self.X[column].dtype):
                X_train[column] = self.X[column].fillna(self.X[column].mean())
            else:
                X_train[column] = self.X[column].fillna('missing')

        columns_missing_values = self.test.columns[self.test.isnull().any()]
        for column in columns_missing_values:
            if is_numeric_dtype(self.test[column].dtype):
                X_test[column] = self.test[column].fillna(self.test[column].mean())
            else:
                X_test[column] = self.test[column].fillna('missing')

        self.categorical = X_train.select_dtypes(include=np.object)
        categories = [list(self.X[c].dropna().unique()) for c in self.categorical]
        for i, c in enumerate(self.categorical):
            categories[i].extend(X_test[c].dropna().unique())
            categories[i].append('missing')

        categories = [sorted(set(c)) for c in categories]


        self.encoder = OrdinalEncoder(categories)
        encoded = self.encoder.fit_transform(self.categorical)

        for i, column in enumerate(self.categorical.columns):
            X_train[column] = encoded[:, i]

        test_categorical = X_test.select_dtypes(include=np.object)
        encoded = self.encoder.transform(test_categorical)
        for i, column in enumerate(test_categorical.columns):
            X_test[column] = encoded[:, i]

        shuffle_columns = [c for c in ['Cabin', 'Name', 'Ticket', 'PassengerId'] if c in X_test.columns]
        X_test[shuffle_columns] = shuffle(X_test[shuffle_columns], random_state=667).reset_index(drop=True)

        self.X_train = X_train
        self.X_test = X_test

        return X_train

    def fit(self):
        if self.X_train is None or len(self.X.columns) < len(self.X_train.columns):
            self.preprocess()

        if self.fitted_model is None:
            self.monotonic = [0] * len(self.X_train.columns)

        estimator = xgboost.XGBClassifier(
            learning_rate=0.1, max_features='sqrt', monotone_constraints=str(tuple(self.monotonic)), random_state=666)

        @use_named_args(self.search_space)
        def objective(**params):
            estimator.set_params(**params)

            scores = cross_val_score(estimator, self.X_train, self.y, cv=5, n_jobs=-1, scoring='roc_auc')
            return -np.mean(scores)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = gp_minimize(
                objective,
                self.search_space,
                n_points=1000,
                n_calls=self.search_iterations,
                n_random_starts=10,
                acq_optimizer='lbfgs',
                n_jobs=-1,
                callback=self._show_progress,
                random_state=666)

        best_params = {f.name: value for (f, value) in zip(self.search_space, result.x)}
        model = xgboost.XGBClassifier(
            learning_rate=0.1, max_features='sqrt', monotone_constraints=str(tuple(self.monotonic)), random_state=666,
            **best_params)
        model.fit(self.X_train, self.y)

        self.fitted_model = model
        return model

    def _show_progress(self, optim_result):
        iteration = len(optim_result.func_vals)
        value = round(-optim_result.fun, 4)
        best_value = round(-min(optim_result.func_vals), 4)
        print("Iteration %s of %s: AUC %s, best AUC: %s" % (iteration, self.search_iterations, value, best_value),
              end='')
        print('\r', end='')

    def drop_feature(self, feature_name):
        self.X.drop(feature_name, axis=1, inplace=True)
        self.test.drop(feature_name, axis=1, inplace=True)

    def evaluate_test(self):
        if self.X_test is None or len(self.test.columns) < len(self.X_test.columns):
            self.preprocess()

        test_probas = self.fitted_model.predict_proba(self.X_test)[:, 1]
        print("AUC of test set: %s" % round(roc_auc_score(self.y_test, test_probas), 4))

    def force_monotonicity(self, feature, direction):
        self.monotonic[self.X_train.columns.get_loc(feature)] = int(direction)

    def feature_importances(self, type):
        xgboost.plot_importance(self.fitted_model, importance_type=type)

    def permutation_importances(self):
        model = deepcopy(self.fitted_model)
        model.fit(self.X_train.as_matrix(), self.y)
        perm = PermutationImportance(model).fit(self.X_train, self.y)

        importances = pd.Series(perm.feature_importances_, index=self.X_train.columns)
        importances /= importances.max()
        importances.sort_values().plot.barh()

    def show_pdp(self, feature_name):
        if self.fitted_model is None:
            self.fit()

        partial_dependence = pdp_isolate(self.fitted_model, self.X_train, self.X_train.columns, feature_name)
        if feature_name == 'Sex':  # encoded feature
            partial_dependence.display_columns = self.encoder.categories_[self.categorical.columns.get_loc('Sex')]

        pdp_plot(partial_dependence, feature_name, center=False, plot_lines=True, x_quantile=True, frac_to_plot=0.2)
