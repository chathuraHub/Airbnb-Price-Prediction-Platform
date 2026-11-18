
import os
import sys
import csv
import io
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.Airbnb.exception import customexception
from src.Airbnb.logger import logging
from src.Airbnb.utils.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler
)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str


class DataTransformation:
    
    def __init__(self, encoding="ordinal"):
        self.encoding = encoding

        self.data_transformation_config = DataTransformationConfig(
            preprocessor_obj_file_path=os.path.join(
                "airbnb-price-prediction",
            f"Preprocessor_{encoding}.pkl"
        )
    )

        

    def get_data_transformation(self):

        try:
            logging.info(
                f"Data Transformation initiated: {self.encoding}"
            )

            numerical_cols = [
                'amenities',
                'accommodates',
                'bathrooms',
                'latitude',
                'longitude',
                'host_response_rate',
                'number_of_reviews',
                'review_scores_rating',
                'bedrooms',
                'beds'
            ]

            categorical_cols = [
                'property_type',
                'room_type',
                'bed_type',
                'cancellation_policy',
                'cleaning_fee',
                'city',
                'host_identity_verified',
                'instant_bookable',
                'host_has_profile_pic'
            ]

            # Categories used by the existing Ordinal Encoder
            property_type_cat = [
                'Apartment', 'House', 'Condominium', 'Townhouse',
                'Loft', 'Other', 'Guesthouse', 'Bed & Breakfast',
                'Bungalow', 'Villa', 'Dorm', 'Guest suite',
                'Camper/RV', 'Timeshare', 'Cabin', 'In-law',
                'Hostel', 'Boutique hotel', 'Boat',
                'Serviced apartment', 'Tent', 'Castle',
                'Vacation home', 'Yurt', 'Hut', 'Treehouse',
                'Chalet', 'Earth House', 'Tipi', 'Train',
                'Cave', 'Casa particular', 'Parking Space',
                'Lighthouse'
            ]

            room_type_cat = [
                'Entire home/apt',
                'Private room',
                'Shared room'
            ]

            bed_type_cat = [
                'Real Bed',
                'Futon',
                'Pull-out Sofa',
                'Airbed',
                'Couch'
            ]

            cancellation_policy_cat = [
                'strict',
                'moderate',
                'flexible',
                'super_strict_30',
                'super_strict_60'
            ]

            cleaning_fee_cat = ['TRUE', 'FALSE']
            city_cat = ['NYC', 'SF', 'DC', 'LA', 'Chicago', 'Boston']

            host_identity_verified_cat = ['t', 'f']
            instant_bookable_cat = ['t', 'f']
            host_has_profile_pic_cat = ['t', 'f']

            logging.info("Building preprocessing pipelines")

            # Numerical pipeline: unchanged
            num_pipeline = Pipeline(
                steps=[
                    (
                        'imputer',
                        SimpleImputer(strategy='median')
                    ),
                    (
                        'scaler',
                        StandardScaler()
                    )
                ]
            )

            # Choose the categorical encoding method
            if self.encoding == "ordinal":

                cat_pipeline = Pipeline(
                    steps=[
                        (
                            'imputer',
                            SimpleImputer(
                                strategy='most_frequent'
                            )
                        ),
                        (
                            'ordinalencoder',
                            OrdinalEncoder(
                                categories=[
                                    property_type_cat,
                                    room_type_cat,
                                    bed_type_cat,
                                    cancellation_policy_cat,
                                    cleaning_fee_cat,
                                    city_cat,
                                    host_identity_verified_cat,
                                    instant_bookable_cat,
                                    host_has_profile_pic_cat
                                ],
                                handle_unknown='use_encoded_value',
                                unknown_value=-1
                            )
                        ),
                        (
                            'scaler',
                            StandardScaler()
                        )
                    ]
                )

            else:
                # One-Hot pipeline: no scaling of binary indicators
                cat_pipeline = Pipeline(
                    steps=[
                        (
                            'imputer',
                            SimpleImputer(
                                strategy='most_frequent'
                            )
                        ),
                        (
                            'onehotencoder',
                            OneHotEncoder(
                                handle_unknown='ignore',
                                sparse_output=False
                            )
                        )
                    ]
                )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'num_pipeline',
                        num_pipeline,
                        numerical_cols
                    ),
                    (
                        'cat_pipeline',
                        cat_pipeline,
                        categorical_cols
                    )
                ],
                remainder='drop'
            )

            return preprocessor

        except Exception as e:
            logging.info(
                "Exception occurred in get_data_transformation"
            )
            raise customexception(e, sys)

    def initialize_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read complete")
            logging.info(
                f"Train DataFrame Head:\n{train_df.head().to_string()}"
            )
            logging.info(
                f"Test DataFrame Head:\n{test_df.head().to_string()}"
            )

            # Normalize cleaning_fee
            train_df['cleaning_fee'] = (
                train_df['cleaning_fee'].astype(str).str.upper()
            )
            test_df['cleaning_fee'] = (
                test_df['cleaning_fee'].astype(str).str.upper()
            )

            # Convert host_response_rate to numeric
            for df in (train_df, test_df):
                df['host_response_rate'] = pd.to_numeric(
                    df['host_response_rate']
                    .astype(str)
                    .str.replace('%', '', regex=False),
                    errors='coerce'
                )

            logging.info(
                "host_response_rate converted to numeric"
            )

            # Target and columns to remove
            target_column_name = 'log_price'

            drop_columns = [
                target_column_name,
                'price',
                'id',
                'name',
                'description',
                'first_review',
                'host_since',
                'last_review',
                'neighbourhood',
                'thumbnail_url',
                'zipcode'
            ]

            # Count amenities while respecting quotation marks
            def count_amenities(value):

                if (
                    pd.isna(value)
                    or str(value).strip() in ("", "{}")
                ):
                    return np.nan

                value = str(value).strip()

                if value.startswith("{") and value.endswith("}"):
                    value = value[1:-1]

                items = next(
                    csv.reader(
                        io.StringIO(value),
                        skipinitialspace=True
                    )
                )

                return sum(
                    1 for item in items if item.strip()
                )

            train_df['amenities'] = (
                train_df['amenities'].apply(count_amenities)
            )
            test_df['amenities'] = (
                test_df['amenities'].apply(count_amenities)
            )

            # Separate input features and target
            input_feature_train_df = train_df.drop(
                columns=drop_columns,
                errors='ignore'
            )

            target_feature_train_df = train_df[
                target_column_name
            ]

            input_feature_test_df = test_df.drop(
                columns=drop_columns,
                errors='ignore'
            )

            target_feature_test_df = test_df[
                target_column_name
            ]

            logging.info(
                f"Encoding selected: {self.encoding}"
            )
            logging.info(
                f"Training input shape: "
                f"{input_feature_train_df.shape}"
            )
            logging.info(
                f"Testing input shape: "
                f"{input_feature_test_df.shape}"
            )

            # Create the selected preprocessor
            preprocessing_obj = self.get_data_transformation()

            # Fit ONLY on training data
            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            # Reuse the fitted preprocessing on test/validation data
            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            logging.info(
                "Preprocessing applied to training and testing data"
            )

            # Append target as the final column
            train_arr = np.concatenate(
                [
                    input_feature_train_arr,
                    np.array(
                        target_feature_train_df
                    ).reshape(-1, 1)
                ],
                axis=1
            )

            test_arr = np.concatenate(
                [
                    input_feature_test_arr,
                    np.array(
                        target_feature_test_df
                    ).reshape(-1, 1)
                ],
                axis=1
            )

            # Save preprocessor
            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            logging.info(
                f"Preprocessor saved using {self.encoding} encoding"
            )

            return train_arr, test_arr

        except Exception as e:
            logging.info(
                "Exception occurred in initialize_data_transformation"
            )
            raise customexception(e, sys)
