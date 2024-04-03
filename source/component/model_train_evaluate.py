import pickle
import pandas as pd
import warnings
from source.logger import logging
from source.exception import CustomException
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

class ModelTrainEvaluate:
    def __init__(self, utility_config):
        self.utility_config = utility_config

        self.models = {
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SVR": SVR(),
            "MLPRegressor": MLPRegressor(),
            "XGBRegressor": XGBRegressor(),
            "LGBMRegressor": LGBMRegressor(),
            "CatBoostRegressor": CatBoostRegressor()
        }

        self.model_evaluation_report = pd.DataFrame(columns=["model_name", "mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "r2_score"])

    # def hyperparameter_tuning(self, x_train, y_train):
    #     try:
    #         model = CatBoostRegressor()
    #
    #         param_grid = {
    #             'learning_rate': [0.01, 0.05, 0.1],
    #             'depth': [4, 6, 8],
    #             'iterations': [100, 200, 300],
    #             'l2_leaf_reg': [1, 3, 5]
    #         }
    #
    #         mse_scorer = make_scorer(mean_squared_error)  # Use mean squared error for scoring
    #
    #         grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=mse_scorer)
    #
    #         grid_search.fit(x_train, y_train)
    #
    #         best_params = grid_search.best_params_
    #         best_score = grid_search.best_score_
    #
    #         return best_params, best_score
    #
    #     except Exception as e:
    #         raise e
    def model_training(self, train_data, test_data):

        try:
            x_train = train_data.drop('selling_price', axis = 1)
            y_train = train_data['selling_price']
            x_test = test_data.drop('selling_price', axis = 1)
            y_test = test_data['selling_price']

            for name, model in self.models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                with open(f"{self.utility_config.model_path}/{name}.pkl", "wb") as f:
                    pickle.dump(model, f)

                self.metrics_and_log(y_test, y_pred, name)

        except CustomException as e:
            raise e

    def metrics_and_log(self, y_test, y_pred, model_name):
        try:
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model: {model_name}, MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R-squared: {r2}")

            new_row = [model_name, mse, rmse, mae, r2]

            self.model_evaluation_report = self.model_evaluation_report._append(pd.Series(new_row, index=self.model_evaluation_report.columns), ignore_index=True)

        except CustomException as e:
            print(e)
            raise e

    def retrain_final_model(self, train_data, test_data):
        try:
            x_train = train_data.drop('selling_price', axis=1)
            y_train = train_data['selling_price']

            x_test = test_data.drop('selling_price', axis=1)
            y_test = test_data['selling_price']

            # best_params, best_score = self.hyperparameter_tuning(x_train, y_train)

            final_model = CatBoostRegressor()
            final_model_name = "CatBoostRegressor"

            final_model.fit(x_train, y_train)

            # Evaluate the model on the test set
            y_pred = final_model.predict(x_test)
            test_score = r2_score(y_test, y_pred)

            logging.info(f"Final model: {CatBoostRegressor}, R2 Score: {test_score}")

            # Save the final model
            with open(f"{self.utility_config.final_model_path}/{final_model_name}.pkl", "wb") as f:
                pickle.dump(final_model, f)

        except CustomException as e:
            raise e

    def initiate_model_training(self):
        try:
            train_data = pd.read_csv(self.utility_config.train_dt_train_file_path+'/'+self.utility_config.train_file_name)
            test_data = pd.read_csv(self.utility_config.train_dt_test_file_path+'/'+self.utility_config.test_file_name)

            self.model_training(train_data, test_data)
            self.model_evaluation_report.to_csv("source/ml/model_evaluation_report.csv", index=False)

            self.retrain_final_model(train_data, test_data)

            print('Model Training Done...')

        except CustomException as e:
            raise e
