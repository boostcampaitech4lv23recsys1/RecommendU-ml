
import os
import numpy as np
import pandas as pd


# CREATE YOUR OWN PREPROCESSOR FOR YOUR MODEL
class Preprocessor():
    def __init__(self):
        self.train_data_path = os.path.join(DATASET_PATH, "train", "train_data", "data")
        self.train_label_path = os.path.join(DATASET_PATH, "train", "train_label")

    def _load_train_dataset(self, train_data_path = None):
        print("starting to load train data: ")
        print(self.train_data_path)
        
        train_data = pd.read_parquet(self.train_data_path) \
            .sort_values(by = ["data_index", "station_seq"], ignore_index = True)
            
        print("train data shape: ")
        print(train_data.shape)
        
        train_label = pd.read_csv(self.train_label_path, header = None, names = LABEL_COLUMNS) \
            .sort_values(by = ["data_index", "station_seq"], ignore_index = True)
            
        print("train label shape: ")
        print(train_label.shape)
        
        return train_data, train_label

    def preprocess_train_dataset(self):
        print("load train data to preprocess...")
        train_data, train_label = self._load_train_dataset()
        
        # windowing to get prev duration
        train_data["prev_ts"] = train_data.groupby("data_index")["ts"].shift(1)
        train_data["prev_ts"] = train_data["prev_ts"].fillna(0)
        train_data["prev_duration"] = np.where(train_data["prev_ts"] == 0, 0, train_data["ts"] - train_data["prev_ts"])

        # drop unnecessary columns
        train_data = train_data[["route_id", "dow", "hour", "prev_duration", "station_lng", "station_lat", "next_station_distance"]]
        train_label = train_label[["next_duration"]]
        print(f"Loaded total data count: {len(train_data)}")

        dataset = tf.data.Dataset.from_tensor_slices((train_data.values, train_label.values)).shuffle(len(train_data))
        return dataset
    
    def preprocess_test_data(self, test_data):
        # windowing to get prev duration
        test_data["prev_ts"] = test_data["ts"].shift(1)
        test_data["prev_ts"] = test_data["prev_ts"].fillna(0)
        test_data["prev_duration"] = np.where(test_data["prev_ts"] == 0, 0, test_data["ts"] - test_data["prev_ts"])
        
        # drop unnecessary columns
        test_data = test_data[["route_id", "dow", "hour", "prev_duration", "station_lng", "station_lat", "next_station_distance"]].tail(1)
        
        dataset = tf.data.Dataset.from_tensor_slices(test_data.values)
        return dataset