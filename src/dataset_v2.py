import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BRSETMultimodalDatasetV2(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. Clean and Engineer Pure Clinical Metadata (NO LEAKAGE)
        self.df['patient_sex'] = (self.df['patient_sex'] == 2).astype(float)
        self.df['diabetes_binary'] = (self.df['diabetes'] == 'yes').astype(float)
        self.df['insuline_binary'] = (self.df['insuline'] == 'yes').astype(float)
        self.df['hypertension'] = self.df['comorbidities'].str.contains('hypertension', case=False, na=False).astype(float)
        
        self.df['diabetes_time_y'] = pd.to_numeric(self.df['diabetes_time_y'], errors='coerce').fillna(0.0)
        self.df['patient_age'] = self.df['patient_age'].fillna(self.df['patient_age'].median())
        
        # Normalize continuous features
        self.df['patient_age'] = (self.df['patient_age'] - self.df['patient_age'].mean()) / (self.df['patient_age'].std() + 1e-8)
        self.df['diabetes_time_y'] = (self.df['diabetes_time_y'] - self.df['diabetes_time_y'].mean()) / (self.df['diabetes_time_y'].std() + 1e-8)
        
        self.tabular_cols = ['patient_age', 'patient_sex', 'diabetes_binary', 'hypertension', 'diabetes_time_y', 'insuline_binary']
        
        # 2. Extract Auxiliary Targets for Multi-Task Learning (Forcing CNN to learn anatomy)
        self.df['optic_disc_abn'] = (self.df['optic_disc'] == '2').astype(float)
        self.df['macula_abn'] = (self.df['macula'] == 2).astype(float)
        self.df['vessels_abn'] = (self.df['vessels'] == 2).astype(float)
        
        # First 5 are Main Tasks, Last 5 are Auxiliary Tasks
        self.target_cols = [
            'diabetic_retinopathy', 'increased_cup_disc', 'macular_edema', 'myopic_fundus', 'amd',
            'drusens', 'hemorrhage', 'optic_disc_abn', 'macula_abn', 'vessels_abn'
        ]
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx]['image_id']) + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        tabular_data = self.df.iloc[idx][self.tabular_cols].values.astype(float)
        tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32)
        
        # Missing tabular features mask
        mask = torch.isnan(tabular_tensor)
        tabular_tensor = torch.nan_to_num(tabular_tensor, nan=0.0)
        
        labels = self.df.iloc[idx][self.target_cols].values.astype(float)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return {
            'image': image,
            'tabular': tabular_tensor,
            'mask': mask.float(),
            'labels': label_tensor
        }