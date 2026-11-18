from src.Airbnb.components.Data_ingestion import DataIngestion
from src.Airbnb.components.Data_transformation import DataTransformation
from src.Airbnb.components.Model_trainer import ModelTrainer

# Data Ingestion: perform this once
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

# Run both encoding experiments on the same train/test split
for encoding in ["ordinal", "onehot"]:

    print(f"\n========== {encoding.upper()} ENCODING ==========\n")

    # Data Transformation
    data_transformation = DataTransformation(encoding=encoding)
    train_arr, test_arr = (
        data_transformation.initialize_data_transformation(
            train_data_path, test_data_path
        )
    )

    # Model Training
    model_trainer_obj = ModelTrainer(encoding=encoding)
    model_trainer_obj.initate_model_training(train_arr, test_arr)