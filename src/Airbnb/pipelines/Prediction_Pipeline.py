import os
import numpy as np
import sys
import pandas as pd
from src.Airbnb.logger import logging
from src.Airbnb.utils.utils import load_object
from src.Airbnb.exception import customexception


class PredictPipeline:
    def __init__(self, encoding="ordinal"):
        self.encoding = encoding

    def predict(self, features):
        try:
            preprocessor_path = os.path.join(
                "airbnb-price-prediction",
                f"Preprocessor_{self.encoding}.pkl"
            )

            model_path = os.path.join(
                "airbnb-price-prediction",
                f"Model_{self.encoding}.pkl"
            )

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            logging.info("Preprocessor and Model Pickle files loaded")

            transformed_data = preprocessor.transform(features)
            pred = model.predict(transformed_data)

            return pred

        except Exception as e:
            raise customexception(e, sys)

class CustomData:
    def __init__(self,
                 property_type: str,
                 room_type: str,
                 amenities: int,
                 accommodates: int,
                 bathrooms: int,
                 bed_type: str,
                 cancellation_policy: str,
                 cleaning_fee: float,
                 city: str,
                 host_has_profile_pic: str,
                 host_identity_verified: str,
                 host_response_rate: str,
                 instant_bookable: str,
                 latitude: float,
                 longitude: float,
                 number_of_reviews: int,
                 review_scores_rating: int,
                 bedrooms: int,
                 beds: int):
        
        self.property_type = property_type
        self.room_type = room_type
        self.amenities = amenities
        self.accommodates = accommodates
        self.bathrooms = bathrooms
        self.bed_type = bed_type
        self.cancellation_policy = cancellation_policy
        self.cleaning_fee = cleaning_fee
        self.city = city
        self.host_has_profile_pic = host_has_profile_pic
        self.host_identity_verified = host_identity_verified
        self.host_response_rate = host_response_rate
        self.instant_bookable = instant_bookable
        self.latitude = latitude
        self.longitude = longitude
        self.number_of_reviews = number_of_reviews
        self.review_scores_rating = review_scores_rating
        self.bedrooms = bedrooms
        self.beds = beds

    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'property_type': [self.property_type],
                'room_type': [self.room_type],
                'amenities': [self.amenities],
                'accommodates': [self.accommodates],
                'bathrooms': [self.bathrooms],
                'bed_type': [self.bed_type],
                'cancellation_policy': [str(self.cancellation_policy).lower()],
                'cleaning_fee': [
                    'TRUE' if str(self.cleaning_fee).strip().upper()
                    in ('1', '1.0', 'TRUE', 'T', 'YES')
                    else 'FALSE'
                ],
                'city': [self.city],
                'host_has_profile_pic': [
                    't' if str(self.host_has_profile_pic).strip().lower()
                    in ('1', '1.0', 'true', 't', 'yes')
                    else 'f'
                ],
                'host_identity_verified': [
                    't' if str(self.host_identity_verified).strip().lower()
                    in ('1', '1.0', 'true', 't', 'yes')
                    else 'f'
                ],
                'host_response_rate': [
                    float(str(self.host_response_rate).replace('%', '').strip())
                    if str(self.host_response_rate).replace('%', '').strip()
                    else 0.0
                ],
                'instant_bookable': [
                    't' if str(self.instant_bookable).strip().lower()
                    in ('1', '1.0', 'true', 't', 'yes')
                    else 'f'
                ],
                'latitude': [self.latitude],
                'longitude': [self.longitude],
                'number_of_reviews': [self.number_of_reviews],
                'review_scores_rating': [self.review_scores_rating],
                'bedrooms': [self.bedrooms],
                'beds': [self.beds]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Prediction input dataframe created and normalized')
            return df

        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise customexception(e, sys)
