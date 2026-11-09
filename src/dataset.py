import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class BRSETMultimodalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, task_type="ssl", is_train=True, modality_dropout_p=0.0):
        self.img_dir = img_dir
        self.transform = transform
        self.task_type = task_type
        self.is_train = is_train
        self.modality_dropout_p = modality_dropout_p
        
        self.df = pd.read_csv(csv_file)
        
        # تشخیص خودکار ستون تصویر
        self.img_col = None
        for col in ['image_id', 'image_path', 'image', 'file_name']:
            if col in self.df.columns:
                self.img_col = col
                break
        if self.img_col is None:
            self.img_col = self.df.columns[0]

        # پیش‌پردازش ایمن ستون‌های پایه
        if 'diabetes' in self.df.columns:
            self.df['diabetes'] = self.df['diabetes'].astype(str).str.lower().str.strip().map(
                lambda x: 1.0 if x in ['yes', '1', '1.0', 'true'] else (0.0 if x in ['no', '0', '0.0', 'false'] else np.nan)
            )
        if 'insuline' in self.df.columns:
            self.df['insuline'] = self.df['insuline'].astype(str).str.lower().str.strip().map(
                lambda x: 1.0 if x in ['yes', '1', '1.0', 'true'] else (0.0 if x in ['no', '0', '0.0', 'false'] else np.nan)
            )
        if 'patient_sex' in self.df.columns:
            self.df['patient_sex'] = self.df['patient_sex'].astype(str).str.lower().str.strip().map(
                lambda x: 1.0 if x in ['1', '1.0', 'm', 'male'] else (0.0 if x in ['2', '2.0', '0', '0.0', 'f', 'female'] else np.nan)
            )
            
        self.df['patient_age'] = pd.to_numeric(self.df.get('patient_age', 0), errors='coerce')
        self.df['diabetes_time_y'] = pd.to_numeric(self.df.get('diabetes_time_y', 0), errors='coerce')
        
        # پیش‌پردازش 3 ویژگی آناتومیک (0 برای نرمال، 1 برای غیرنرمال)
        for col in ['macula', 'optic_disc', 'vessels']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower().apply(lambda x: 0.0 if 'normal' in x else 1.0)
            else:
                self.df[col] = 0.0
        
        self.tabular_cols = [
            'patient_age', 'patient_sex', 'diabetes', 'diabetes_time_y', 'insuline',
            'macula', 'optic_disc', 'vessels'
        ]
        self.tabular_data = self.df[self.tabular_cols].copy()
        
        # ماسک داده‌های موجود و جایگذاری میانگین برای داده‌های گمشده
        self.missing_mask = (~self.tabular_data.isna()).astype(np.float32).values
        col_means = self.tabular_data.mean().fillna(0.0)
        self.tabular_data = self.tabular_data.fillna(col_means).values.astype(np.float32)
        
        # نرمال‌سازی Z-score
        self.scaler = StandardScaler()
        self.tabular_data = self.scaler.fit_transform(self.tabular_data).astype(np.float32)
        
        # مپینگ یکپارچه و استاندارد 3 کلاسه (0: Normal, 1: NPDR [1,2,3], 2: PDR [4])
        self.labels = []
        for val in self.df['DR_ICDR'].values:
            val_int = int(val)
            if val_int == 0:
                self.labels.append(0)
            elif val_int in [1, 2, 3]:
                self.labels.append(1)
            else:
                self.labels.append(2)
        self.labels = np.array(self.labels, dtype=np.int64)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx][self.img_col]).strip()
        if not any(img_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            img_name = f"{img_name}.jpg"
            
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        tab_features = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        missing_mask = torch.tensor(self.missing_mask[idx], dtype=torch.float32)
        
        # تفکیک هوشمند Dropout بین فاز SSL و فاز Fine-tuning
        if self.is_train and self.modality_dropout_p > 0.0:
            if self.task_type == "ssl":
                feat_drop_mask = (torch.rand_like(tab_features) > self.modality_dropout_p).float()
                if feat_drop_mask.sum() == 0:
                    feat_drop_mask[torch.randint(0, len(tab_features), (1,))] = 1.0
                tab_features = tab_features * feat_drop_mask
                missing_mask = missing_mask * feat_drop_mask
            else:
                if torch.rand(1).item() < self.modality_dropout_p:
                    tab_features = torch.zeros_like(tab_features)
                    missing_mask = torch.zeros_like(missing_mask)
                
        label = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return {
            'image': image,
            'tabular': tab_features,
            'mask': missing_mask,
            'labels': label
        }