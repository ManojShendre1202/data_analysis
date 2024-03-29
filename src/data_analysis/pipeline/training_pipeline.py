from data_analysis.components.EDA import *
from data_analysis.components.model_training import ModelTrainer
from data_analysis.components.data_ingestion import DataIngestion
from data_analysis.components.data_prepration import DataTransformation

def main():
    logging.info("Starting EDA Process")

    # # Training Pipeline
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.Start_data_ingestion()

    # #transformation pipeline
    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transormation(train_data_path,test_data_path)

    # model training
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)
    # print(model_trainer.initiate_model_trainer(train_arr,test_arr))
if __name__ == "__main__":
    main()