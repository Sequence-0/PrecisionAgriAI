import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import FHB_Dataset, normHSI

def train_and_save_model():
    # Load dataset
    dataset = FHB_Dataset(data_dir="E:/College/beyond-visible-spectrum-ai-for-agriculture-2024/ICPR01/kaggle", transform=None, noisy=False, rotate=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Perform spectral averaging
    spec_data = []
    for data in dataloader:
        img, label = data
        img = normHSI(img).squeeze(0)
        spec = []
        for i in range(img.shape[0]):
            mean = torch.mean(img[i,:,:])
            spec.append(mean.item())
        spec.append(label.item())
        spec_data.append(spec)
    
    df = pd.DataFrame(spec_data, columns=[*range(101), 'label'])

    # Train initial model to get feature importance
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'num_leaves': 30,
        'learning_rate': 0.015,
        'verbose': -1
    }
    
    model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=300)
    
    # Get top 30 features based on gain importance
    gain_importance = model.feature_importance(importance_type='gain')
    gain_importance_df = pd.DataFrame({'Band No.': range(101), 'Gain': gain_importance})
    top_30_features = gain_importance_df.sort_values(by='Gain', ascending=False)['Band No.'][:30].tolist()
    
    print("Top 30 Feature Indices:", top_30_features)
    
    # Train final model on top 30 features
    X_final = df.drop('label', axis=1)[top_30_features]
    y_final = df['label']
    
    X_train_final, _, y_train_final, _ = train_test_split(X_final, y_final, test_size=0.1, random_state=123)
    
    final_train_data = lgb.Dataset(X_train_final, label=y_train_final)
    
    final_model = lgb.train(params, final_train_data, num_boost_round=300)
    
    # Save the final model
    final_model.save_model('model.txt')
    print("Model saved as model.txt")

if __name__ == '__main__':
    train_and_save_model()
