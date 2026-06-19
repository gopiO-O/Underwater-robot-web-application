"""
Flood Risk Prediction - Machine Learning Model Training
Uses Random Forest Classifier for flood prediction based on sensor data
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_and_explore_data(filepath):
    """Load dataset and display basic information"""
    print("=" * 70)
    print("LOADING FLOOD PREDICTION DATASET")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Samples: {df.shape[0]}")
    
    print("\nFeature Names:")
    for i, col in enumerate(df.columns[:-1], 1):
        print(f"  {i}. {col}")
    
    print(f"\nTarget Variable: {df.columns[-1]}")
    print(f"\nClass Distribution:")
    print(df['flood_risk'].value_counts())
    print(f"\nPercentage:")
    print(df['flood_risk'].value_counts(normalize=True) * 100)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    return df


def prepare_data(df):
    """Split data into features and target, then train/test sets"""
    print("\n" + "=" * 70)
    print("PREPARING DATA FOR TRAINING")
    print("=" * 70)
    
    # Separate features (X) and target (y)
    # IMPORTANT: Don't use flood_risk_score as a feature - it's the calculated risk!
    X = df.drop(['flood_risk', 'flood_risk_score'], axis=1)
    y = df['flood_risk']
    
    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    print(f"\nTraining set class distribution:")
    print(y_train.value_counts())
    
    print(f"\nTesting set class distribution:")
    print(y_test.value_counts())
    
    return X_train, X_test, y_train, y_test, X.columns


def train_model(X_train, y_train):
    """Train Random Forest Classifier"""
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    # Random Forest is excellent for this task because:
    # 1. Handles non-linear relationships well
    # 2. Resistant to overfitting
    # 3. Provides feature importance
    # 4. No need for feature scaling
    # 5. Works well with imbalanced datasets
    
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        max_depth=15,          # Maximum depth of trees
        min_samples_split=10,  # Minimum samples required to split
        min_samples_leaf=5,    # Minimum samples at leaf node
        random_state=42,
        n_jobs=-1,            # Use all CPU cores
        class_weight='balanced' # Handle class imbalance
    )
    
    print("\nModel Parameters:")
    print(model.get_params())
    
    print("\n[*] Training model...")
    model.fit(X_train, y_train)
    print("[+] Model training completed!")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Evaluate model performance"""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"\n[*] Training Accuracy: {train_accuracy * 100:.2f}%")
    
    # Testing accuracy
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"[*] Testing Accuracy: {test_accuracy * 100:.2f}%")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n[*] Cross-Validation Scores: {cv_scores}")
    print(f"[*] Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)")
    
    # Detailed classification report
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION REPORT (Test Set)")
    print("=" * 70)
    print("\n" + classification_report(y_test, test_pred, 
                                       target_names=['No Flood Risk', 'Flood Risk']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_pred)
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                  No Flood  |  Flood")
    print("         --------+----------+----------")
    print(f"No Flood |       {cm[0][0]:4d}   |   {cm[0][1]:4d}")
    print(f"Flood    |       {cm[1][0]:4d}   |   {cm[1][1]:4d}")
    
    # ROC-AUC Score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n[*] ROC-AUC Score: {roc_auc:.4f}")
    
    # Feature Importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n")
    for idx, row in feature_importance.iterrows():
        bar = '█' * int(row['importance'] * 100)
        print(f"{row['feature']:25s} | {bar} {row['importance']:.4f}")
    
    return test_accuracy, feature_importance


def save_model(model, filename='flood_prediction_model.pkl'):
    """Save trained model to disk"""
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    joblib.dump(model, filename)
    print(f"\n[+] Model saved to: {filename}")
    print(f"[*] Model size: {round(os.path.getsize(filename) / 1024, 2)} KB")
    
    return filename


def test_predictions(model, feature_names):
    """Test model with sample predictions"""
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    
    # Test scenarios based on risk levels
    test_cases = [
        {
            'name': '✅ Safe Conditions (Score: 0-30)',
            'water_level_height_cm': 25.0,
            'rainfall_forecast_mm': 5.0,
            'air_humidity_percent': 50.0,
            'water_turbidity_ntu': 80.0,
            'trend_rising_6h_cm': -2.0
        },
        {
            'name': '⚠️ Low Risk (Score: 31-50)',
            'water_level_height_cm': 45.0,
            'rainfall_forecast_mm': 25.0,
            'air_humidity_percent': 65.0,
            'water_turbidity_ntu': 220.0,
            'trend_rising_6h_cm': 3.0
        },
        {
            'name': '🟠 Moderate Risk (Score: 51-70) - DANGER!',
            'water_level_height_cm': 65.0,
            'rainfall_forecast_mm': 45.0,
            'air_humidity_percent': 75.0,
            'water_turbidity_ntu': 380.0,
            'trend_rising_6h_cm': 7.0
        },
        {
            'name': '🔴 High Risk (Score: 71-85) - EVACUATE!',
            'water_level_height_cm': 75.0,
            'rainfall_forecast_mm': 90.0,
            'air_humidity_percent': 88.0,
            'water_turbidity_ntu': 550.0,
            'trend_rising_6h_cm': 12.0
        },
        {
            'name': '⛔ CRITICAL (Score: 86-100) - IMMEDIATE ACTION!',
            'water_level_height_cm': 85.0,
            'rainfall_forecast_mm': 150.0,
            'air_humidity_percent': 95.0,
            'water_turbidity_ntu': 780.0,
            'trend_rising_6h_cm': 16.0
        }
    ]
    
    for case in test_cases:
        name = case.pop('name')
        # Create DataFrame with correct column order
        X_sample = pd.DataFrame([case])[feature_names]
        
        prediction = model.predict(X_sample)[0]
        probability = model.predict_proba(X_sample)[0]
        
        print(f"\n{'='*70}")
        print(f"📋 {name}")
        print(f"{'='*70}")
        print(f"   Water Height: {case['water_level_height_cm']} cm {'🔴 DANGER!' if case['water_level_height_cm'] > 60 else ''}")
        print(f"   Rainfall: {case['rainfall_forecast_mm']} mm")
        print(f"   Humidity: {case['air_humidity_percent']}%")
        print(f"   Turbidity: {case['water_turbidity_ntu']} NTU")
        print(f"   Trend (6h): {case['trend_rising_6h_cm']:+.1f} cm {'↑ Rising' if case['trend_rising_6h_cm'] > 0 else '↓ Falling'}")
        print(f"   {'─'*66}")
        print(f"   ➜ Prediction: {'🔴 FLOOD RISK' if prediction == 1 else '✅ NO FLOOD'}")
        print(f"   ➜ Confidence: No Flood {probability[0]*100:.1f}% | Flood {probability[1]*100:.1f}%")


def main():
    """Main training pipeline"""
    import os
    
    # File path
    data_file = 'flood_prediction_dataset.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found!")
        print("Please run generate_flood_dataset.py first")
        return
    
    # Step 1: Load and explore data
    df = load_and_explore_data(data_file)
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
    
    # Step 3: Train model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate model
    test_accuracy, feature_importance = evaluate_model(
        model, X_train, X_test, y_train, y_test, feature_names
    )
    
    # Step 5: Save model
    model_file = save_model(model)
    
    # Step 6: Test with sample predictions
    test_predictions(model, feature_names)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\n[+] Model: Random Forest Classifier")
    print(f"[+] Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"[+] Model saved: {model_file}")
    print(f"\n[*] Most Important Feature: {feature_importance.iloc[0]['feature']}")
    print(f"[+] Ready for deployment!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
