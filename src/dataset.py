import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class BRSETMultimodalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, task_type="ssl"):
        """
        Multimodal dataset compatible with labels_brset.csv
        task_type options: 'ssl', 'multilabel_5', 'dr_3class', 'dr_binary'
        """
        self.img_dir = img_dir
        self.transform = transform
        self.task_type = task_type
        
        self.df = pd.read_csv(csv_file)
        
        # Preprocess Tabular columns
        self.df['diabetes'] = self.df['diabetes'].map({'yes': 1, 'no': 0})
        self.df['insuline'] = self.df['insuline'].map({'yes': 1, 'no': 0})
        self.df['diabetes_time_y'] = pd.to_numeric(self.df['diabetes_time_y'], errors='coerce')
        
        # Extract features for Contrastive alignment
        self.tabular_cols = ['patient_age', 'patient_sex', 'diabetes', 'diabetes_time_y', 'insuline']
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
        # 1. Image
        img_id = str(self.df.iloc[idx]['image_id'])
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # 2. Tabular and Mask
        tab_features = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        missing_mask = torch.tensor(self.missing_mask[idx], dtype=torch.float32)
        
        # 3. Dynamic Label Extraction
        if self.task_type == "multilabel_5":
            target_cols = ['diabetic_retinopathy', 'increased_cup_disc', 'macular_edema', 'myopic_fundus', 'amd']
            label = torch.tensor(self.df.iloc[idx][target_cols].values.astype(float), dtype=torch.float32)
            
        elif self.task_type == "dr_3class":
            dr_raw = int(self.df.iloc[idx]['DR_ICDR'])
            # Mapping: 0->Normal(0), 1,2->NPDR(1), 3,4->PDR(2)
            if dr_raw == 0:
                mapped_grade = 0
            elif dr_raw in [1, 2]:
                mapped_grade = 1
            else:
                mapped_grade = 2
            label = torch.tensor(mapped_grade, dtype=torch.long)
            
        elif self.task_type == "dr_binary":
            has_dr = float(self.df.iloc[idx]['diabetic_retinopathy'])
            label = torch.tensor([has_dr], dtype=torch.float32)
            
        else:
            # Default fallback for SSL pre-training
            label = torch.tensor(self.df.iloc[idx]['DR_ICDR'], dtype=torch.long)
            
        return {
            'image': image,
            'tabular': tab_features,
            'mask': missing_mask,
            'labels': label
        }

if __name__ == "__main__":
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = BRSETMultimodalDataset(
        csv_file='../data/raw/labels_brset.csv', 
        img_dir='../data/raw/images/', 
        transform=transform,
        task_type="multilabel_5"
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    
    print("Batch Check:")
    print("Image:", batch['image'].shape)
    print("Tabular:", batch['tabular'].shape)
    print("Labels:", batch['labels'].shape) # Should be [16, 5] for multilabel_5