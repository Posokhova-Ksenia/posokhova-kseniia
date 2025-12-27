import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import lightgbm as lgb
import optuna
from scipy import stats
import joblib


class OptimizedDynamicPricing:
    def __init__(self, data_path='./data', n_folds=5, use_optuna=True, n_trials=20):
        self.data_path = Path(data_path)
        self.n_folds = n_folds
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.models_p05 = []
        self.models_p95 = []
        self.best_params_p05 = None
        self.best_params_p95 = None
        self.label_encoders = {}
        self.feature_importance = None

    def load_and_analyze(self):
        """Загрузка данных с правильной проверкой дат"""
        train = pd.read_csv(self.data_path / 'train.csv', parse_dates=['dt'])
        test = pd.read_csv(self.data_path / 'test.csv', parse_dates=['dt'])
        sample_sub = pd.read_csv(self.data_path / 'sample_submission.csv')

        print(f"Train: {train.shape}, Test: {test.shape}")

        # Проверяем временные интервалы
        print(f"Train date range: {train['dt'].min()} to {train['dt'].max()}")
        print(f"Test date range: {test['dt'].min()} to {test['dt'].max()}")

        train_dates = set(train['dt'].dt.date)
        test_dates = set(test['dt'].dt.date)
        common_dates = train_dates.intersection(test_dates)

        if len(common_dates) > 0:
            print(f"ВНИМАНИЕ: {len(common_dates)} общих дат в train и test!")
        else:
            print("Train и test не имеют общих дат")

        # Проверяем продукты
        train_products = set(train['product_id'].unique())
        test_products = set(test['product_id'].unique())
        common_products = train_products.intersection(test_products)

        print(f"\nАНАЛИЗ ПРОДУКТОВ:")
        print(f"Уникальных продуктов в train: {len(train_products)}")
        print(f"Уникальных продуктов в test: {len(test_products)}")
        print(
            f"Общих продуктов: {len(common_products)} ({len(common_products) / len(test_products) * 100:.1f}% от test)")

        return train, test, sample_sub

    def create_time_features(self, df):
        df = df.copy()

        # Базовые временные признаки
        df['year'] = df['dt'].dt.year
        df['month'] = df['dt'].dt.month
        df['day'] = df['dt'].dt.day
        df['dayofweek'] = df['dt'].dt.dayofweek
        df['dayofyear'] = df['dt'].dt.dayofyear
        df['week'] = df['dt'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['dt'].dt.quarter
        df['dayofmonth'] = df['dt'].dt.day

        # Циклические признаки
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

        # Флаги
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['dt'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['dt'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['dt'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['dt'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df['dt'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['dt'].dt.is_year_end.astype(int)

        return df

    def create_lag_features_safe(self, df, is_train=True, target_columns=None):
        df = df.copy()

        if target_columns is None:
            target_columns = []

        if not is_train or not target_columns:
            return df

        df = df.sort_values(['product_id', 'dt'])

        for product_id in df['product_id'].unique():
            product_mask = df['product_id'] == product_id
            idx = df[product_mask].index

            if len(idx) > 7:
                for target_col in target_columns:
                    if target_col in df.columns:
                        # Лаги на 1, 2, 3, 7, 14 дней
                        for lag in [1, 2, 3, 7, 14]:
                            if lag < len(idx):
                                df.loc[idx[lag:], f'{target_col}_lag_{lag}'] = (
                                    df.loc[idx[:-lag], target_col].values
                                )

                        # Разности лагов
                        for lag1, lag2 in [(1, 2), (1, 3), (1, 7)]:
                            col1 = f'{target_col}_lag_{lag1}'
                            col2 = f'{target_col}_lag_{lag2}'
                            if col1 in df.columns and col2 in df.columns:
                                df[f'{target_col}_diff_{lag1}_{lag2}'] = df[col1] - df[col2]

                        # Скользящие статистики
                        for window in [3, 7, 14, 28]:
                            if window < len(idx):
                                # Expanding window
                                exp_mean = df.loc[idx, target_col].expanding(min_periods=2).mean()
                                exp_std = df.loc[idx, target_col].expanding(min_periods=2).std()
                                exp_min = df.loc[idx, target_col].expanding(min_periods=2).min()
                                exp_max = df.loc[idx, target_col].expanding(min_periods=2).max()

                                df.loc[idx, f'{target_col}_exp_mean_{window}'] = exp_mean.shift(1)
                                df.loc[idx, f'{target_col}_exp_std_{window}'] = exp_std.shift(1)
                                df.loc[idx, f'{target_col}_exp_min_{window}'] = exp_min.shift(1)
                                df.loc[idx, f'{target_col}_exp_max_{window}'] = exp_max.shift(1)

        return df

    def create_interaction_features(self, df):
        df = df.copy()

        numerical_interactions = [
            ('n_stores', 'avg_temperature'),
            ('n_stores', 'avg_humidity'),
            ('avg_temperature', 'avg_humidity'),
            ('avg_temperature', 'precpt'),
            ('avg_humidity', 'precpt'),
            ('n_stores', 'precpt')
        ]

        for col1, col2 in numerical_interactions:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

        for col in ['avg_temperature', 'avg_humidity', 'precpt', 'n_stores']:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_log'] = np.log1p(df[col])

        if 'dayofweek' in df.columns:
            df['is_monday'] = (df['dayofweek'] == 0).astype(int)
            df['is_friday'] = (df['dayofweek'] == 4).astype(int)

        if 'month' in df.columns:
            df['is_december'] = (df['month'] == 12).astype(int)
            df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
            df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

        return df

    def create_aggregate_features(self, df, is_train=True, train_df=None):
        df = df.copy()

        group_cols = [
            'management_group_id', 'first_category_id',
            'second_category_id', 'third_category_id',
            'dayofweek', 'month'
        ]

        value_cols = ['n_stores', 'avg_temperature', 'avg_humidity', 'precpt']

        if is_train:
            for group in group_cols:
                if group in df.columns:
                    for value in value_cols:
                        if value in df.columns:
                            stats_df = df.groupby(group)[value].agg([
                                'mean', 'std', 'min', 'max', 'median', 'skew'
                            ]).add_prefix(f'{value}_{group}_')

                            for stat_col in stats_df.columns:
                                df[stat_col] = df[group].map(stats_df[stat_col])
        else:
            if train_df is not None:
                for group in group_cols:
                    if group in df.columns and group in train_df.columns:
                        for value in value_cols:
                            if value in df.columns and value in train_df.columns:
                                stats_df = train_df.groupby(group)[value].agg([
                                    'mean', 'std', 'min', 'max', 'median', 'skew'
                                ]).add_prefix(f'{value}_{group}_')

                                for stat_col in stats_df.columns:
                                    df[stat_col] = df[group].map(stats_df[stat_col])

        return df

    def prepare_features(self, train_df, test_df):

        train_processed = self.create_time_features(train_df)
        test_processed = self.create_time_features(test_df)

        if 'price_p05' in train_processed.columns and 'price_p95' in train_processed.columns:
            train_processed['price_mid'] = (train_processed['price_p05'] + train_processed['price_p95']) / 2
            train_processed['price_width'] = train_processed['price_p95'] - train_processed['price_p05']

            print("Создание лаговых признаков для train...")
            train_processed = self.create_lag_features_safe(
                train_processed,
                is_train=True,
                target_columns=['price_mid', 'price_width', 'price_p05', 'price_p95']
            )

        train_processed = self.create_aggregate_features(train_processed, is_train=True)
        test_processed = self.create_aggregate_features(test_processed, is_train=False, train_df=train_processed)

        train_processed = self.create_interaction_features(train_processed)
        test_processed = self.create_interaction_features(test_processed)

        lag_columns = [col for col in train_processed.columns
                       if any(x in col for x in ['_lag_', '_exp_', '_diff_'])]

        for lag_col in lag_columns:
            if lag_col in train_processed.columns:
                last_values = train_processed.groupby('product_id')[lag_col].last().to_dict()
                test_processed[lag_col] = test_processed['product_id'].map(last_values)

        categorical_cols = [
            'management_group_id',
            'first_category_id',
            'second_category_id',
            'third_category_id'
        ]

        for col in categorical_cols:
            if col in train_processed.columns:
                le = LabelEncoder()
                train_processed[f'{col}_encoded'] = le.fit_transform(train_processed[col].astype(str))
                test_processed[f'{col}_encoded'] = le.transform(test_processed[col].astype(str))
                self.label_encoders[col] = le

        for df in [train_processed, test_processed]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    if col in train_processed.columns:
                        fill_val = train_processed[col].median()
                    else:
                        fill_val = 0
                    df[col] = df[col].fillna(fill_val)

            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols:
                df[col] = df[col].fillna('missing')

        return train_processed, test_processed

    def select_features(self, train_df, test_df):

        exclude_cols = [
            'dt', 'product_id', 'row_id',
            'price_p05', 'price_p95', 'price_mid', 'price_width'
        ]

        original_cat_cols = [
            'management_group_id', 'first_category_id',
            'second_category_id', 'third_category_id'
        ]

        exclude_cols.extend(original_cat_cols)

        all_features = [col for col in train_df.columns if col not in exclude_cols]

        missing_in_test = [col for col in all_features if col not in test_df.columns]
        if missing_in_test:
            all_features = [col for col in all_features if col not in missing_in_test]

        variance_threshold = 0.01
        low_variance_features = []

        for col in all_features:
            if train_df[col].dtype != 'object':
                variance = train_df[col].var()
                if variance < variance_threshold:
                    low_variance_features.append(col)

        if low_variance_features:
            all_features = [col for col in all_features if col not in low_variance_features]

        print(f"Используем {len(all_features)} признаков")

        return all_features

    def objective_p05(self, trial, X_train, y_train, X_val, y_val):
        params = {
            'objective': 'quantile',
            'alpha': 0.05,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'random_state': 322,
            'n_jobs': -1,
            'verbose': -1
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='quantile',
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        val_pred = model.predict(X_val)
        mae = np.mean(np.abs(y_val - val_pred))

        return mae

    def objective_p95(self, trial, X_train, y_train, X_val, y_val):
        """Функция для оптимизации гиперпараметров p95"""
        params = {
            'objective': 'quantile',
            'alpha': 0.95,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'random_state': 322,
            'n_jobs': -1,
            'verbose': -1
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='quantile',
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        val_pred = model.predict(X_val)
        mae = np.mean(np.abs(y_val - val_pred))

        return mae

    def optimize_hyperparameters(self, X_train, y_train_p05, y_train_p95):

        tscv = TimeSeriesSplit(n_splits=3)

        study_p05 = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=322),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )

        train_idx, val_idx = list(tscv.split(X_train))[0]
        X_opt_train = X_train.iloc[train_idx]
        X_opt_val = X_train.iloc[val_idx]
        y_opt_train_p05 = y_train_p05[train_idx]
        y_opt_val_p05 = y_train_p05[val_idx]

        study_p05.optimize(
            lambda trial: self.objective_p05(
                trial, X_opt_train, y_opt_train_p05, X_opt_val, y_opt_val_p05
            ),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params_p05 = study_p05.best_params

        study_p95 = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=322),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )

        y_opt_train_p95 = y_train_p95[train_idx]
        y_opt_val_p95 = y_train_p95[val_idx]

        study_p95.optimize(
            lambda trial: self.objective_p95(
                trial, X_opt_train, y_opt_train_p95, X_opt_val, y_opt_val_p95
            ),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params_p95 = study_p95.best_params

        return self.best_params_p05, self.best_params_p95

    def get_default_params(self):
        """Параметры по умолчанию если не используем оптимизацию"""
        params_p05 = {
            'objective': 'quantile',
            'alpha': 0.05,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'num_leaves': 63,
            'max_depth': 7,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_split_gain': 0.01,
            'subsample_freq': 5,
            'random_state': 322,
            'n_jobs': -1,
            'verbose': -1
        }

        params_p95 = params_p05.copy()
        params_p95['alpha'] = 0.95

        return params_p05, params_p95

    def train_models(self, X_train, y_train_p05, y_train_p95):

        if self.use_optuna:
            params_p05, params_p95 = self.optimize_hyperparameters(
                X_train, y_train_p05, y_train_p95
            )
        else:
            params_p05, params_p95 = self.get_default_params()

        # TimeSeriesSplit для кросс-валидации
        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        # Для OOF
        oof_p05 = np.zeros(len(X_train))
        oof_p95 = np.zeros(len(X_train))

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):

            # Разделяем
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]

            y_tr_p05 = y_train_p05[train_idx]
            y_val_p05 = y_train_p05[val_idx]
            y_tr_p95 = y_train_p95[train_idx]
            y_val_p95 = y_train_p95[val_idx]

            # Модель для p05
            model_p05 = lgb.LGBMRegressor(**params_p05)
            model_p05.fit(
                X_tr, y_tr_p05,
                eval_set=[(X_val, y_val_p05)],
                eval_metric='quantile',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=100),
                    lgb.log_evaluation(period=100)
                ]
            )

            model_p95 = lgb.LGBMRegressor(**params_p95)
            model_p95.fit(
                X_tr, y_tr_p95,
                eval_set=[(X_val, y_val_p95)],
                eval_metric='quantile',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=100),
                    lgb.log_evaluation(period=100)
                ]
            )

            # Предсказания
            val_pred_p05 = model_p05.predict(X_val)
            val_pred_p95 = model_p95.predict(X_val)

            # Постобработка
            val_pred_p05, val_pred_p95 = self.postprocess_predictions(
                val_pred_p05, val_pred_p95, y_tr_p05, y_tr_p95
            )

            # Сохраняем OOF
            oof_p05[val_idx] = val_pred_p05
            oof_p95[val_idx] = val_pred_p95

            # Сохраняем модели
            self.models_p05.append(model_p05)
            self.models_p95.append(model_p95)

            # Метрики
            fold_iou = self.calculate_iou(
                y_val_p05, y_val_p95, val_pred_p05, val_pred_p95
            )
            fold_scores.append(fold_iou)

            # Анализ ширины
            true_width = y_val_p95 - y_val_p05
            pred_width = val_pred_p95 - val_pred_p05

            # Сохраняем важность признаков с первого фолда
            if fold == 1:
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance_p05': model_p05.feature_importances_,
                    'importance_p95': model_p95.feature_importances_
                })
                importance_df['importance_mean'] = importance_df[['importance_p05', 'importance_p95']].mean(axis=1)
                self.feature_importance = importance_df.sort_values('importance_mean', ascending=False)

        # Итоги CV
        cv_mean = np.mean(fold_scores)
        cv_std = np.std(fold_scores)

        print(f"\n{'=' * 60}")
        print("ИТОГИ CV:")
        print(f"Средний IoU: {cv_mean:.6f} (±{cv_std:.6f})")
        print(f"Лучший фолд: {max(fold_scores):.6f}")
        print(f"Худший фолд: {min(fold_scores):.6f}")
        print('=' * 60)

        # Анализ OOF
        self.analyze_oof_predictions(oof_p05, oof_p95, y_train_p05, y_train_p95)

        return cv_mean, oof_p05, oof_p95

    def postprocess_predictions(self, pred_p05, pred_p95, train_p05=None, train_p95=None):

        # 1. Гарантируем минимальную ширину
        widths = pred_p95 - pred_p05
        min_width = 0.001

        # Если есть тренировочные данные, вычисляем разумную минимальную ширину
        if train_p05 is not None and train_p95 is not None:
            train_widths = train_p95 - train_p05
            min_width = max(min_width, np.percentile(train_widths[train_widths > 0], 5))

        mask_narrow = widths < min_width
        if mask_narrow.any():
            for i in np.where(mask_narrow)[0]:
                mid = (pred_p05[i] + pred_p95[i]) / 2
                pred_p05[i] = mid - min_width / 2
                pred_p95[i] = mid + min_width / 2

        # 2. Гарантируем, что p95 > p05
        mask_invalid = pred_p05 >= pred_p95
        if mask_invalid.any():
            for i in np.where(mask_invalid)[0]:
                if pred_p05[i] > pred_p95[i]:
                    pred_p05[i], pred_p95[i] = pred_p95[i], pred_p05[i]
                # Еще раз проверяем ширину
                if pred_p95[i] - pred_p05[i] < min_width:
                    mid = (pred_p05[i] + pred_p95[i]) / 2
                    pred_p05[i] = mid - min_width / 2
                    pred_p95[i] = mid + min_width / 2

        # 3. Ограничиваем диапазон значений
        # Основано на анализе данных
        pred_p05 = np.clip(pred_p05, 0.5, 1.5)
        pred_p95 = np.clip(pred_p95, 0.6, 1.6)

        # 4. Гарантируем, что p95 > p05 + минимальная ширина
        final_mask = pred_p05 >= pred_p95
        if final_mask.any():
            for i in np.where(final_mask)[0]:
                pred_p95[i] = pred_p05[i] + min_width

        return pred_p05, pred_p95

    def calculate_iou(self, true_p05, true_p95, pred_p05, pred_p95, epsilon=1e-8):
        # Пересечение
        intersection = np.maximum(
            0,
            np.minimum(true_p95, pred_p95) - np.maximum(true_p05, pred_p05)
        )

        # Объединение
        union = (true_p95 - true_p05) + (pred_p95 - pred_p05) - intersection

        # IoU (избегаем деления на 0)
        iou = intersection / np.maximum(union, epsilon)

        return np.mean(iou)

    def analyze_oof_predictions(self, oof_p05, oof_p95, true_p05, true_p95):

        iou = self.calculate_iou(true_p05, true_p95, oof_p05, oof_p95)

        true_width = true_p95 - true_p05
        pred_width = oof_p95 - oof_p05

        # Распределение IoU
        iou_per_sample = self.calculate_iou_per_sample(true_p05, true_p95, oof_p05, oof_p95)

        print(f"\nРаспределение IoU по наблюдениям:")
        percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
        for p in percentiles:
            val = np.percentile(iou_per_sample, p)
            print(f"  {p:3d}%: {val:.6f}")

        # Процент хороших предсказаний
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        for thresh in thresholds:
            ratio = (iou_per_sample > thresh).mean() * 100
            print(f"Процент > {thresh}: {ratio:.1f}%")

    def calculate_iou_per_sample(self, true_p05, true_p95, pred_p05, pred_p95, epsilon=1e-8):
        intersection = np.maximum(
            0,
            np.minimum(true_p95, pred_p95) - np.maximum(true_p05, pred_p05)
        )

        union = (true_p95 - true_p05) + (pred_p95 - pred_p05) - intersection

        return intersection / np.maximum(union, epsilon)

    def predict(self, X_test):

        if not self.models_p05 or not self.models_p95:
            raise ValueError("Модели не обучены!")

        # Ансамблирование предсказаний от всех фолдов
        all_p05 = []
        all_p95 = []

        for model_p05, model_p95 in zip(self.models_p05, self.models_p95):
            pred_p05 = model_p05.predict(X_test)
            pred_p95 = model_p95.predict(X_test)

            all_p05.append(pred_p05)
            all_p95.append(pred_p95)

        # Усреднение
        ensemble_p05 = np.mean(all_p05, axis=0)
        ensemble_p95 = np.mean(all_p95, axis=0)

        final_p05, final_p95 = self.postprocess_predictions(ensemble_p05, ensemble_p95)

        # Проверка
        widths = final_p95 - final_p05
        n_invalid = (final_p05 >= final_p95).sum()

        return final_p05, final_p95

    def save_models(self, path='models'):
        import os
        os.makedirs(path, exist_ok=True)

        for i, (model_p05, model_p95) in enumerate(zip(self.models_p05, self.models_p95)):
            joblib.dump(model_p05, f'{path}/model_p05_fold_{i}.pkl')
            joblib.dump(model_p95, f'{path}/model_p95_fold_{i}.pkl')

        print(f"Модели сохранены в папку {path}")

    def run_pipeline(self):
        import time
        start_time = time.time()

        train, test, sample_sub = self.load_and_analyze()

        train_processed, test_processed = self.prepare_features(train, test)

        selected_features = self.select_features(train_processed, test_processed)

        # 4. Подготовка данных
        X_train = train_processed[selected_features]
        y_train_p05 = train_processed['price_p05'].values
        y_train_p95 = train_processed['price_p95'].values

        X_test = test_processed[selected_features]

        cv_score, oof_p05, oof_p95 = self.train_models(
            X_train, y_train_p05, y_train_p95
        )

        test_p05, test_p95 = self.predict(X_test)

        submission = sample_sub.copy()
        submission['price_p05'] = test_p05
        submission['price_p95'] = test_p95

        # Дополнительная проверка
        widths = test_p95 - test_p05
        invalid_ratio = (test_p05 >= test_p95).mean()

        print(f"\n ИТОГОВАЯ ПРОВЕРКА SUBMISSION:")
        print(f"Всего строк: {len(submission)}")
        print(f"Средняя ширина: {widths.mean():.6f}")
        print(f"Медианная ширина: {np.median(widths):.6f}")
        print(f"Некорректных интервалов: {invalid_ratio * 100:.2f}%")
        print(f"p05 статистика: mean={test_p05.mean():.4f}, std={test_p05.std():.4f}")
        print(f"p95 статистика: mean={test_p95.mean():.4f}, std={test_p95.std():.4f}")

        # Сохраняем submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'results/submission_optimized_{timestamp}.csv'
        submission.to_csv(output_file, index=False)

        self.save_models(f'models_{timestamp}')

        if self.feature_importance is not None:
            feat_file = f'feature_importance_{timestamp}.csv'
            self.feature_importance.to_csv(feat_file, index=False)

            print(self.feature_importance.head(20)[['feature', 'importance_mean']])

        # 9. Итоги
        elapsed = time.time() - start_time

        print(f"\n{'=' * 80}")
        print(f" ПАЙПЛАЙН ЗАВЕРШЕН ЗА {elapsed / 60:.1f} МИНУТ")
        print(f" CV IoU: {cv_score:.6f}")

        return submission, cv_score


# ЗАПУСК
if __name__ == "__main__":
    data_path = Path('./data')

    # Проверка файлов
    required = ['train.csv', 'test.csv', 'sample_submission.csv']
    missing = [f for f in required if not (data_path / f).exists()]

    model = OptimizedDynamicPricing(
        data_path='./data',
        n_folds=7,
        use_optuna=True,
        n_trials=10
    )

    submission, score = model.run_pipeline()
