from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


def preprocess_data(data, target_column, save_path, file_path, target_encoder_path):
    # hapus data duplikat
    data = data.drop_duplicates()
       
    # Menentukan fitur numerik dan kategoris
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
    column_names = data.columns
    column_names = data.columns.drop(target_column)
    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)
    
    # Menyimpan nama kolom sebgai header tanpa data
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")
    
    # Pastikan target_column tidak ada di numeric_features atau categorical features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
        
    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder())
    ])    
    
    
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Memisahkan target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fitting dan transformasi data pada training set
    X_train = preprocessor.fit_transform(X_train)
    # Transformasi pada data testing
    X_test = preprocessor.transform(X_test)
    
    # Simpan pipeline
    dump(preprocessor, save_path)
    
    # Encoder untuk target
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train)
    y_test = target_encoder.transform(y_test)
    dump(target_encoder, target_encoder_path)
    
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = numeric_features + categorical_features
        
    # dataframe fitur hasil transformasi
    X_train = pd.DataFrame(X_train, columns=feature_names)
    
    X_train[target_column] = y_train
    X_train.to_csv(file_path, index=False)
    
    return X_train, X_test, y_train, y_test
    
 

# data = pd.read_csv("personality_dataset.csv")
# X_train, X_test, y_train, y_test = preprocess_data(data, 
#                                                    'Personality', 
#                                                    'preprocessor_pipeline.joblib', 
#                                                    'preprocessing/data_preprocessing.csv',
#                                                    'target_encoder.joblib')

