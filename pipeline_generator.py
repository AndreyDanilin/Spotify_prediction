#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 21

def load_and_prepare_data():
    
    df_60 = pd.read_csv("data/dataset-of-60s.csv")
    df_60['decade_of_release'] = '60'
    
    df_70 = pd.read_csv("data/dataset-of-70s.csv")
    df_70['decade_of_release'] = '70'
    
    df_80 = pd.read_csv("data/dataset-of-80s.csv")
    df_80['decade_of_release'] = '80'
    
    df_90 = pd.read_csv("data/dataset-of-90s.csv")
    df_90['decade_of_release'] = '90'
    
    df_00 = pd.read_csv("data/dataset-of-00s.csv")
    df_00['decade_of_release'] = '0'
    
    df_10 = pd.read_csv("data/dataset-of-10s.csv")
    df_10['decade_of_release'] = '10'
    
    df = pd.concat([df_00, df_10, df_60, df_70, df_80, df_90])

    df = df[['uri','track','artist','danceability','energy','key','loudness',
             'mode','speechiness','acousticness','instrumentalness','liveness',
             'valence','tempo','duration_ms','time_signature','chorus_hit',
             'sections','decade_of_release','target']]
    
    return df

def create_features_and_target(df):

    

    df_merge = df.drop_duplicates(subset=['uri'])
    X = df_merge.drop(['uri', 'track', 'target'], axis=1)
    y = df_merge['target']
    
    
    return X, y

def create_preprocessor(X):

    categorical_cols = ['artist', 'decade_of_release']
    text_cols = ['track']
    numerical_cols = [col for col in X.columns if col not in categorical_cols and col not in text_cols]
    
    preprocessor = ColumnTransformer([
        ('artist_encoder', TargetEncoder(), ['artist']),
        ('decade_encoder', OneHotEncoder(drop='first', sparse_output=False), ['decade_of_release']),
        ('scaler', StandardScaler(), numerical_cols)
    ], remainder='passthrough')
    
    return preprocessor

def create_xgb_model():
    
    xgb_model = xgb.XGBClassifier(
        learning_rate=0.028929893320248787,
        max_depth=10,
        subsample=0.5388792823570937,
        colsample_bytree=0.33367343724613546,
        min_child_weight=1,
        random_state=RANDOM_STATE
    )
    
    return xgb_model

def train_pipeline(X, y, preprocessor, xgb_model):
   
    xgb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])
    
    xgb_pipe.fit(X, y)
    return xgb_pipe

def save_pipeline(xgb_pipe, X, y, filename="xgb_pipe.joblib"):
    xgb_pipe.fit(X, y)
    
    # Сохранение
    joblib.dump(xgb_pipe, filename)
    print(f"Pipeline was saved in {filename}")

def main():
    try:
        df = load_and_prepare_data()
        X, y = create_features_and_target(df)
        preprocessor = create_preprocessor(X)
        xgb_model = create_xgb_model()
        xgb_pipe = train_pipeline(X, y, preprocessor, xgb_model)
        save_pipeline(xgb_pipe, X, y)
        print("Pipeline created")
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()
