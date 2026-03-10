import glob
from datetime import datetime
import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import neurokit2 as nk
import sklearn.preprocessing as skp
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

data_dir = r'..\half-life-alyx-sessions'

session_1 = []
session_2 = []

def read_session_match_file():
    # Open the file
    with open('session_match.txt', 'r') as file:
        # Read each line
        for line in file:
            # Split the line by comma
            values = line.strip().split(',')
            # Assign the values to session_1 and session_2
            session_1.append(int(values[0]))
            session_2.append(int(values[1]))
    file.close()
    print(len(session_1), session_1)
    print(len(session_2), session_2)
            
read_session_match_file()

# Define device
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Function to load ECG data dynamically skipping intermediate folder
def load_ecg_from_folders_with_labels(base_path, subject_pairs, window_size=300):
    train_data_list = []
    test_data_list = []
    train_label_list = []
    test_label_list = []
    
    for label, (train_subject, test_subject) in enumerate(subject_pairs):
        # Handle both train and test subjects in the pair
        for subject in [train_subject, test_subject]:
            # Dynamically construct the path with a wildcard for the intermediate folder
            search_pattern = os.path.join(base_path, f"{subject}", "body-captures", "*", "ECG.json","PPG.json")
            file_paths = glob.glob(search_pattern)  # Get all matching paths
            print(file_paths)
            if not file_paths:
                print(f"No matching file for subject {subject} at {search_pattern}")
                continue

            file_path = file_paths[0]  # Take the first matching file
            with open(file_path, 'r') as f:
                data_dict = json.load(f)
            data = [val[0] for key, val in data_dict['data'].items()]
            data = np.array(data, dtype=np.float32)
            print(data.shape)

            # Split data into windows of 250 samples
            num_windows = len(data) // window_size
            windows = data[:num_windows * window_size].reshape(-1, window_size)

            if subject == train_subject:
                # Append data and labels
                train_data_list.append(windows)
                print("train_data_shape: ", len(windows), windows)
                train_label_list.append(np.full(windows.shape[0], label, dtype=np.int64))
                print("train_label_shape: ", len(np.full(windows.shape[0], label, dtype=np.int64)), np.full(windows.shape[0], label, dtype=np.int64))
            
            if subject == test_subject:
                # Append data and labels
                test_data_list.append(windows)
                print("test_data_shape: ", len(windows), windows)
                test_label_list.append(np.full(windows.shape[0], label, dtype=np.int64))
                print("test_label_shape: ", len(np.full(windows.shape[0], label, dtype=np.int64)), np.full(windows.shape[0], label, dtype=np.int64))

        
    if train_data_list and test_data_list:
        return np.concatenate(train_data_list), np.concatenate(train_label_list), np.concatenate(test_data_list), np.concatenate(test_label_list)
    else:
        return np.empty((0, window_size)), np.empty((0,))
    

class ECGDataset(Dataset):
    def __init__(self, ecg_data, ppg_data):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data

    def __getitem__(self, index):

        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        
        window_size = ecg.shape[-1]

        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        ecg = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=128, method="pantompkins1985")
        _, info = nk.ecg_peaks(ecg, sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)

        # Create a numpy array for ROI regions with the same shape as ECG
        ecg_roi_array = np.zeros_like(ecg.reshape(1, window_size))

        # Iterate through ECG R peaks and set values to 1 within the ROI regions
        roi_size = 32
        for peak in info["ECG_R_Peaks"]:
            roi_start = max(0, peak - roi_size // 2)
            roi_end = min(roi_start + roi_size, window_size)
            ecg_roi_array[0, roi_start:roi_end] = 1

        return ecg.reshape(1, window_size).copy(), ppg.reshape(1, window_size).copy(), ecg_roi_array.copy() #, ppg_cwt.copy()

    def __len__(self):
        return len(self.ecg_data)
def get_datasets(
    DATA_PATH = "../../datasets/", 
    datasets=["BIDMC", "CAPNO", "DALIA", "MIMIC-AFib", "WESAD"],
    window_size=4,
    ):

    ecg_train_list = []
    ppg_train_list = []
    ecg_test_list = []
    ppg_test_list = []
    
    for dataset in datasets:

        ecg_train = np.load(DATA_PATH + dataset + f"/ecg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        ppg_train = np.load(DATA_PATH + dataset + f"/ppg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        
        ecg_test = np.load(DATA_PATH + dataset + f"/ecg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
        ppg_test = np.load(DATA_PATH + dataset + f"/ppg_test_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)

        ecg_train_list.append(ecg_train)
        ppg_train_list.append(ppg_train)
        ecg_test_list.append(ecg_test)
        ppg_test_list.append(ppg_test)

    ecg_train = np.nan_to_num(np.concatenate(ecg_train_list).astype("float32"))
    ppg_train = np.nan_to_num(np.concatenate(ppg_train_list).astype("float32"))

    ecg_test = np.nan_to_num(np.concatenate(ecg_test_list).astype("float32"))
    ppg_test = np.nan_to_num(np.concatenate(ppg_test_list).astype("float32"))

    dataset_train = ECGDataset(
        skp.minmax_scale(ecg_train, (-1, 1), axis=1),
        skp.minmax_scale(ppg_train, (-1, 1), axis=1)
    )
    dataset_test = ECGDataset(
        skp.minmax_scale(ecg_test, (-1, 1), axis=1),
        skp.minmax_scale(ppg_test, (-1, 1), axis=1)
    )

    return dataset_train, dataset_test

# Paths and subject pairs
base_path = "/sessions"
train_subjects = [1, 3, 6, 7, 8, 9, 10, 12, 14, 18, 21, 22, 23, 27, 28, 29, 30, 33, 34, 35, 40, 41, 43, 44, 55, 56, 57, 61, 63, 64, 65]
test_subjects = [2, 4, 13, 15, 16, 17, 19, 24, 26, 42, 53, 32, 31, 37, 38, 50, 39, 45, 46, 48, 47, 49, 52, 51, 60, 58, 68, 62, 69, 66, 67]

# Pair train and test subjects and assign a unique label to each pair
subject_pairs = list(zip(train_subjects, test_subjects))
print(subject_pairs)

# Load all data and assign labels
data, labels, test_data, test_labels = load_ecg_from_folders_with_labels(base_path, subject_pairs)
print(data.shape, labels.shape)
