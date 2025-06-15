from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


def preprocess_data(data, target_column, save_path, file_path):
    # Menentukan fitur numerik dan kategoris
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    column_names = data.columns
    # Mengambil nama kolom tanpa target
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
    
    return X_train, X_test, y_train, y_test 


# Validasi memastikan tahapan preprocessing dilakukan dan disimpan dengan baik
def inference(new_data, load_path):
    # Memuat pipeline preprocessing
    preprocessor = load(load_path)
    print(f"Pipeline preprocessing dimuat dari: {load_path}")
    
    # Transformasi data baru
    transformed_data = preprocessor.transform(new_data)
    return transformed_data


data = pd.read_csv("personality_dataset.csv")
data.head()

X_train, X_test, y_train, y_test = preprocess_data(data, 'Personality', 'preprocessor_pipeline.joblib', 'data.csv')