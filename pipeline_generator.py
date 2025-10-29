# pipeline_generator.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from category_encoders import TargetEncoder
from sentence_transformers import SentenceTransformer

RANDOM_STATE = 21

def normalize_decade(value):
    try:
        val = int(value)
        if 1960 <= val < 1970:
            return '60'
        elif 1970 <= val < 1980:
            return '70'
        elif 1980 <= val < 1990:
            return '80'
        elif 1990 <= val < 2000:
            return '90'
        elif 2000 <= val < 2010:
            return '0'
        elif 2010 <= val < 2020:
            return '10'
        else:
            return 'unknown'
    except Exception:
        return str(value)

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

def add_track_embeddings(df, track_col='track', model_name='all-MiniLM-L6-v2'):
    """Добавляем эмбеддинги трека как фичи"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[track_col].astype(str).values, show_progress_bar=True, batch_size=32)
    # embeddings: np.ndarray shape (N, D)
    for i in range(embeddings.shape[1]):
        df[f'track_emb_{i}'] = embeddings[:, i]
    df = df.drop(columns=[track_col])
    return df

def preprocess_df(df):
    """Единая функция предобработки для train/infer"""
    # Приведение десятилетий
    df['decade_of_release'] = df['decade_of_release'].apply(normalize_decade)
    # Убираем пропуски, типы
    if 'artist' not in df.columns:
        df['artist'] = 'unknown_artist'
    if 'decade_of_release' not in df.columns:
        df['decade_of_release'] = 'unknown_decade'
    df['artist'] = df['artist'].astype(str)
    df['decade_of_release'] = df['decade_of_release'].astype(str)
    for col in df.columns:
        if col not in ['artist', 'decade_of_release', 'uri', 'target']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0.0)
    return df

def create_features_and_target(df):
    df_merge = df.drop_duplicates(subset=['uri'])
    X = df_merge.drop(['uri', 'target'], axis=1)
    y = df_merge['target']
    return X, y

def create_preprocessor(X):
    categorical_cols = ['artist', 'decade_of_release']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
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

def save_pipeline(xgb_pipe, filename="xgb_pipe.joblib"):
    joblib.dump(xgb_pipe, filename)
    print(f"Pipeline was saved in {filename}")

def main():
    try:
        df = load_and_prepare_data()
        df = preprocess_df(df)
        df = add_track_embeddings(df)
        X, y = create_features_and_target(df)
        preprocessor = create_preprocessor(X)
        xgb_model = create_xgb_model()
        xgb_pipe = train_pipeline(X, y, preprocessor, xgb_model)
        save_pipeline(xgb_pipe)
        print("Pipeline created")
    except Exception as e:
        print(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()
