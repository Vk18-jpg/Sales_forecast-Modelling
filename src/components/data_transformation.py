import os
import sys
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
     preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.scaler = StandardScaler()

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # Load train & test data
            logging.info("Reading merged data...")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            # Make copies for feature engineering
            data_train = df_train.copy()
            data_test = df_test.copy()

            # --- Feature Engineering ---
            logging.info("Performing feature engineering...")
            for data in [data_train, data_test]:
                # Convert Date to datetime
                data["Date"] = pd.to_datetime(data["Date"])

                # Days to holidays
                data["Days_to_Thanksgiving"] = (pd.to_datetime("2012-11-22") - data["Date"]).dt.days
                data["Days_to_Christmas"] = (pd.to_datetime("2012-12-25") - data["Date"]).dt.days

                # Holiday flags
                data["SuperBowlWeek"] = data["Date"].between("2010-02-01", "2010-02-07") | \
                                        data["Date"].between("2011-02-07", "2011-02-13") | \
                                        data["Date"].between("2012-01-30", "2012-02-05")

                data["LaborDay"] = data["Date"].between("2010-09-06", "2010-09-06") | \
                                   data["Date"].between("2011-09-05", "2011-09-05") | \
                                   data["Date"].between("2012-09-03", "2012-09-03")

                data["Thanksgiving"] = data["Date"].between("2010-11-25", "2010-11-25") | \
                                       data["Date"].between("2011-11-24", "2011-11-24") | \
                                       data["Date"].between("2012-11-22", "2012-11-22")

                data["Christmas"] = data["Date"].between("2010-12-25", "2010-12-25") | \
                                    data["Date"].between("2011-12-25", "2011-12-25") | \
                                    data["Date"].between("2012-12-25", "2012-12-25")

                # Markdowns Sum
                markdown_cols = [col for col in data.columns if "MarkDown" in col]
                if markdown_cols:
                    data["MarkdownsSum"] = data[markdown_cols].sum(axis=1)

                # Fill missing values
                # data.fillna(0, inplace=True)

                # Encode categorical features
                if "IsHoliday" in data.columns:
                    data["IsHoliday"] = data["IsHoliday"].astype(int)
                if "Type" in data.columns:
                    data["Type"] = data["Type"].map({"A": 1, "B": 2, "C": 3}).fillna(0).astype(int)

            data_train.fillna(0, inplace = True)
            data_test['CPI'].fillna(data_test['CPI'].mean(), inplace = True)
            data_test['Unemployment'].fillna(data_test['Unemployment'].mean(), inplace = True)
            data_test.fillna(0, inplace = True)


            # Define features & target
            X_train = data_train.drop(columns=["Weekly_Sales", "Date"])
            y_train = data_train["Weekly_Sales"]

            X_test = data_test.drop(columns=["Date"])
            

                
            logging.info("Scaling features with StandardScaler...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            

            # Save transformed data
            logging.info("Saving preprocessor object...")
            # with open(self.config.preprocessor_obj_file_path, "wb") as f:
            #     pickle.dump(self.scaler, f)

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=self.scaler

            )

            return (
                train_arr,
                X_test_scaled,
                
                self.config.preprocessor_obj_file_path,)

        except Exception as e:
            raise CustomException(e, sys)
