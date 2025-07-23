import os
import sys
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import ( AdaBoostRegressor, GradientBoostingRegressor,
                               RandomForestRegressor )

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and testing input data')
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'linear regression': LinearRegression(),
                'k-neighbors': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False),
                'AdaBoost': AdaBoostRegressor()
            }

            # model_report: dict = evaluate_model(x_train, y_train, x_test, y_test, models,models=models)
            # ...existing code...
            model_report: dict = evaluate_model(x_train, y_train, x_test, y_test, models)
            # ...existing code...

            best_model_name = None
            best_model_score = float('-inf')
            for model_name, scores in model_report.items():
                if scores['test_score'] > best_model_score:
                    best_model_score = scores['test_score']
                    best_model_name = model_name

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f'Best model found: {best_model_name} with score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted) 
            return r2_square


        except Exception as e:
            raise CustomException(e, sys)



            