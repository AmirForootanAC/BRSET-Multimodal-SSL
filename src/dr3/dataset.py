import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class BRSETMultimodalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, task_type="ssl"):
        self.img_dir = img_dir
        self.transform = transform
        self.task_type = task_type
        
        self.df = pd.read_csv(csv_file)
        
        # Preprocess 5 Base Tabular columns
        self.df['diabetes'] = self.df['diabetes'].map({'yes': 1, 'no': 0})
        self.df['insuline'] = self.df['insuline'].map({'yes': 1, 'no': 0})
        self.df['diabetes_time_y'] = pd.to_numeric(self.df['diabetes_time_y'], errors='coerce')
        
        # Preprocess 3 Anatomical features (0 for normal, 1 for abnormal/pathological)
        for col in ['macula', 'optic_disc', 'vessels']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower()
                self.df[col] = self.df[col].apply(lambda x: 0 if 'normal' in x else 1)
        
        # Total 8 features for Contrastive alignment
        self.tabular_cols = [
            'patient_age', 'patient_sex', 'diabetes', 'diabetes_time_y', 'insuline',
            'macula', 'optic_disc', 'vessels'
        ]
        self.tabular_data = self.df[self.tabular_cols].copy()
        
        # Missing mask and Imputation
        self.missing_mask = (~self.tabular_data.isna()).astype(float).values
        self.tabular_data = self.tabular_data.fillna(self.tabular_data.mean()).values
        
        # Normalization (Z-score)
        self.scaler = StandardScaler()
        self.tabular_data = self.scaler.fit_transform(self.tabular_data)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = str(self.df.iloc[idx]['image_id'])
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        tab_features = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        missing_mask = torch.tensor(self.missing_mask[idx], dtype=torch.float32)
        
        if self.task_type == "dr_3class":
            dr_raw = int(self.df.iloc[idx]['DR_ICDR'])
            # Mapping: 0->Normal(0), 1,2->NPDR(1), 3,4->PDR(2)
            if dr_raw == 0:
                mapped_grade = 0
            elif dr_raw in [1, 2]:
                mapped_grade = 1
            else:
                mapped_grade = 2
            label = torch.tensor(mapped_grade, dtype=torch.long)
            
        else:
            # Fallback for SSL pre-training
            label = torch.tensor(self.df.iloc[idx]['DR_ICDR'], dtype=torch.long)
            
        return {
            'image': image,
            'tabular': tab_features,
            'mask': missing_mask,
            'labels': label
        }