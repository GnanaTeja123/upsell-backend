import joblib
import pandas as pd
import json
import io
from datetime import datetime
import os

# --- Load Models and the CRUCIAL Preprocessor Artifact ---
SERVICE_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(SERVICE_FILE_DIR, '..', '..', '..', 'ml', 'saved_models'))

try:
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    classifier = joblib.load(os.path.join(MODEL_DIR, 'classifier.pkl'))
    segmenter = joblib.load(os.path.join(MODEL_DIR, 'segmenter.pkl'))
    with open(os.path.join(MODEL_DIR, 'performance_metrics.json'), 'r') as f:
        performance_metrics = json.load(f)
    print("✅ All models and artifacts loaded successfully.")
except Exception as e:
    print(f"🔴 Error loading artifacts: {e}. Please run the training script again.")
    preprocessor, classifier, segmenter, performance_metrics = None, None, None, None

# In-memory storage for raw and processed data
raw_data_store = {}
processed_data_store = {}

def process_and_store_data(df: pd.DataFrame, file_id: str):
    if not all([preprocessor, classifier, segmenter]):
        raise RuntimeError("Models or preprocessor not loaded. Check backend logs.")
    
    raw_data_store[file_id] = df.copy() # Store the original data

    predict_df = df.copy()
    predict_df['last_purchase_date'] = pd.to_datetime(predict_df['last_purchase_date'])
    predict_df['days_since_last_purchase'] = (datetime.now() - predict_df['last_purchase_date']).dt.days
    X_predict = predict_df.drop(['customer_id', 'last_purchase_date', 'upsell_accepted'], axis=1, errors='ignore')

    X_processed = preprocessor.transform(X_predict)

    probabilities = classifier.predict_proba(X_processed)[:, 1]
    segments = segmenter.predict(X_processed)
    segment_map = {0: 'Medium Potential', 1: 'Low Potential', 2: 'High Potential'}
    
    df['upsellLikelihood'] = probabilities
    df['estimatedValue'] = df['upsellLikelihood'] * df['avg_purchase_value']
    df['segment'] = [segment_map.get(s, 'Unknown') for s in segments]
    processed_data_store[file_id] = df
    print(f"✅ Data for file_id '{file_id}' processed and stored.")

def get_summary_metrics(file_id: str):
    df = processed_data_store.get(file_id)
    if df is None: raise ValueError("Processed data not found.")
    high_potential_count = (df['segment'] == 'High Potential').sum()
    revenue_forecast = (df[df['segment'] == 'High Potential']['estimatedValue']).sum()
    return {"totalCustomers": int(len(df)), "highPotentialCount": int(high_potential_count), "estimatedRevenue": float(revenue_forecast)}

def get_segmentation_data(file_id: str):
    df = processed_data_store.get(file_id)
    if df is None: raise ValueError("Processed data not found.")
    segment_counts = df['segment'].value_counts().reset_index()
    segment_counts.columns = ['name', 'value']
    return segment_counts.to_dict(orient='records')

def get_high_potential_customers(file_id: str):
    df = processed_data_store.get(file_id)
    if df is None: raise ValueError("Processed data not found.")
    return df[df['segment'] == 'High Potential'].sort_values(by='upsellLikelihood', ascending=False).head(100).to_dict(orient='records')

def get_customer_demographics(file_id: str):
    df = raw_data_store.get(file_id)
    if df is None: raise ValueError("Raw data not found.")
    return {
        "gender_dist": df['gender'].value_counts().reset_index().to_dict(orient='records'),
        "income_dist": df['income_level'].value_counts().reset_index().to_dict(orient='records'),
        "location_dist": df['location'].value_counts().nlargest(10).reset_index().to_dict(orient='records')
    }

    s
def get_dataset_distributions(file_id: str):
    df = raw_data_store.get(file_id)
    if df is None: raise ValueError("Raw data not found.")
    satisfaction_dist = df['satisfaction_score'].value_counts().sort_index().reset_index()
    satisfaction_dist.columns = ['satisfaction_score', 'count']
    return { "description": df.describe().to_dict(), "satisfaction_dist": satisfaction_dist.to_dict(orient='records') }
    
def get_model_performance():
    if performance_metrics is None: raise RuntimeError("Performance metrics not loaded.")
    return performance_metrics

def get_customer_analysis(file_id: str):
    df = raw_data_store.get(file_id)
    if df is None: raise ValueError("Raw data not found.")
    
    # Age distribution
    age_bins = [18, 30, 45, 60, 100]
    age_labels = ['18-30', '31-45', '46-60', '60+']
    age_dist = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).value_counts().sort_index().reset_index()
    age_dist.columns = ['age_group', 'count']

    # Tenure distribution
    tenure_bins = [0, 24, 60, 120, df['tenure'].max() + 1]
    tenure_labels = ['0-2 years', '2-5 years', '5-10 years', '10+ years']
    tenure_dist = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, right=False).value_counts().sort_index().reset_index()
    tenure_dist.columns = ['tenure_group', 'count']
    
    return {
        "gender_dist": df['gender'].value_counts().reset_index().to_dict(orient='records'),
        "income_dist": df['income_level'].value_counts().reset_index().to_dict(orient='records'),
        "age_dist": age_dist.to_dict(orient='records'),
        "tenure_dist": tenure_dist.to_dict(orient='records')
    }