import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from data_analysis.utils.helper import save_object
from data_analysis.utils.exception import CustomException
from data_analysis.utils.logger import logging
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('models','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_columns = ["year", "mileage","tax","mpg","engineSize"]
            categorical_columns = [
                "transmission",
                "fuelType",
                "Manufacturer"
            ]
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())

            ])
            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]

            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transormation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'price'

            if target_column_name not in train_df.columns:
                raise ValueError(f'Target column "{target_column_name}" not present in the train dataset.')

            if target_column_name not in test_df.columns:
                raise ValueError(f'Target column "{target_column_name}" not present in the test dataset.')

            target_feature_train_df = train_df[target_column_name]
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)

            ## divide the test dataset to independent and dependent feature
            target_feature_test_df = test_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(sys, e)