import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime

def train_and_save_models():
    print("🚀 Starting robust model training process...")

    # --- 1. Load Data ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'raw', 'synthetic_customer_dataset.csv'))
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("🔴 CRITICAL ERROR: 'synthetic_customer_dataset.csv' not found in 'ml/data/raw/'.")
        return
    print("✅ Data loaded successfully.")
    
    # --- 2. Feature Engineering & Data Split ---
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    df['days_since_last_purchase'] = (datetime.now() - df['last_purchase_date']).dt.days
    X = df.drop(['customer_id', 'last_purchase_date', 'upsell_accepted'], axis=1)
    y = df['upsell_accepted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 3. Define and Fit the Preprocessor ---
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough')

    print("Fitting the data preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # --- 4. Train Models on PROCESSED Data ---
    print("💪 Training classification model...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    classifier.fit(X_train_processed, y_train)

    print("📊 Evaluating model performance...")
    y_pred = classifier.predict(X_test_processed)
    performance_metrics = {
        "accuracy": accuracy_score(y_test, y_pred), "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred), "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    print("💪 Training clustering model...")
    # Process the full dataset with the already-fitted preprocessor
    X_full_processed = preprocessor.transform(X)
    segmenter = KMeans(n_clusters=3, random_state=42, n_init=10)
    segmenter.fit(X_full_processed)

    # --- 5. Save All Artifacts ---
    model_dir = os.path.normpath(os.path.join(script_dir, '..', 'saved_models'))
    os.makedirs(model_dir, exist_ok=True)
    
    # CRUCIAL: Save the preprocessor itself
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    joblib.dump(classifier, os.path.join(model_dir, 'classifier.pkl'))
    joblib.dump(segmenter, os.path.join(model_dir, 'segmenter.pkl'))
    
    with open(os.path.join(model_dir, 'performance_metrics.json'), 'w') as f:
        json.dump(performance_metrics, f, indent=4)
        
    print(f"✅ Preprocessor and models saved successfully in '{model_dir}'")

if __name__ == '__main__':
    train_and_save_models()