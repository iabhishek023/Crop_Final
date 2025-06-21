import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class CropRecommendationModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
    def create_sample_data(self):
        """Create sample crop recommendation data"""
        np.random.seed(42)
        n_samples = 2000
        
        # Define crop characteristics (approximate values)
        crops_data = {
            'rice': {'N': (80, 120), 'P': (40, 80), 'K': (40, 80), 'temp': (20, 35), 'humidity': (80, 95), 'ph': (5.5, 7.0), 'rainfall': (150, 300)},
            'wheat': {'N': (50, 100), 'P': (30, 60), 'K': (30, 70), 'temp': (12, 25), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (50, 100)},
            'corn': {'N': (120, 180), 'P': (60, 100), 'K': (40, 80), 'temp': (18, 35), 'humidity': (55, 75), 'ph': (5.8, 7.2), 'rainfall': (60, 110)},
            'cotton': {'N': (100, 160), 'P': (40, 80), 'K': (50, 100), 'temp': (21, 35), 'humidity': (50, 80), 'ph': (5.8, 8.0), 'rainfall': (50, 100)},
            'sugarcane': {'N': (150, 250), 'P': (50, 100), 'K': (150, 250), 'temp': (26, 35), 'humidity': (70, 90), 'ph': (6.0, 8.5), 'rainfall': (100, 150)},
            'potato': {'N': (80, 120), 'P': (50, 80), 'K': (120, 180), 'temp': (15, 25), 'humidity': (80, 95), 'ph': (4.8, 5.8), 'rainfall': (50, 70)},
            'tomato': {'N': (100, 150), 'P': (40, 80), 'K': (150, 200), 'temp': (20, 30), 'humidity': (60, 80), 'ph': (6.0, 7.0), 'rainfall': (60, 90)},
            'banana': {'N': (100, 200), 'P': (75, 150), 'K': (300, 600), 'temp': (26, 35), 'humidity': (75, 85), 'ph': (5.5, 7.0), 'rainfall': (100, 180)},
            'apple': {'N': (60, 120), 'P': (25, 50), 'K': (100, 200), 'temp': (10, 25), 'humidity': (60, 80), 'ph': (6.0, 7.0), 'rainfall': (100, 125)},
            'orange': {'N': (80, 120), 'P': (30, 60), 'K': (100, 150), 'temp': (15, 30), 'humidity': (55, 75), 'ph': (6.0, 7.5), 'rainfall': (100, 120)}
        }
        
        data = []
        crops = list(crops_data.keys())
        samples_per_crop = n_samples // len(crops)
        
        for crop in crops:
            params = crops_data[crop]
            for _ in range(samples_per_crop):
                sample = {
                    'N': np.random.normal((params['N'][0] + params['N'][1])/2, (params['N'][1] - params['N'][0])/6),
                    'P': np.random.normal((params['P'][0] + params['P'][1])/2, (params['P'][1] - params['P'][0])/6),
                    'K': np.random.normal((params['K'][0] + params['K'][1])/2, (params['K'][1] - params['K'][0])/6),
                    'temperature': np.random.normal((params['temp'][0] + params['temp'][1])/2, (params['temp'][1] - params['temp'][0])/6),
                    'humidity': np.random.normal((params['humidity'][0] + params['humidity'][1])/2, (params['humidity'][1] - params['humidity'][0])/6),
                    'ph': np.random.normal((params['ph'][0] + params['ph'][1])/2, (params['ph'][1] - params['ph'][0])/6),
                    'rainfall': np.random.normal((params['rainfall'][0] + params['rainfall'][1])/2, (params['rainfall'][1] - params['rainfall'][0])/6),
                    'label': crop
                }
                # Add some noise and ensure realistic bounds
                sample['N'] = max(0, sample['N'])
                sample['P'] = max(0, sample['P'])
                sample['K'] = max(0, sample['K'])
                sample['temperature'] = max(-10, min(50, sample['temperature']))
                sample['humidity'] = max(0, min(100, sample['humidity']))
                sample['ph'] = max(0, min(14, sample['ph']))
                sample['rainfall'] = max(0, sample['rainfall'])
                
                data.append(sample)
        
        return pd.DataFrame(data)
    
    def load_data(self, file_path=None):
        """Load data from file or create sample data"""
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            print("Creating sample dataset...")
            df = self.create_sample_data()
            # Save sample data
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/crop_recommendation.csv', index=False)
            print("Sample data saved to data/crop_recommendation.csv")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        # Separate features and target
        X = df[self.feature_names]
        y = df['label']
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_encoded
    
    def train_model(self, file_path=None):
        """Train the crop recommendation model"""
        # Load data
        df = self.load_data(file_path)
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # Detailed evaluation
        crop_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=crop_names))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return accuracy
    
    def save_model(self, model_dir='models'):
        """Save the trained model and preprocessors"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, f'{model_dir}/crop_model.joblib')
        joblib.dump(self.scaler, f'{model_dir}/scaler.joblib')
        joblib.dump(self.label_encoder, f'{model_dir}/label_encoder.joblib')
        
        print(f"Model saved to {model_dir}/")
    
    def load_model(self, model_dir='models'):
        """Load the trained model and preprocessors"""
        self.model = joblib.load(f'{model_dir}/crop_model.joblib')
        self.scaler = joblib.load(f'{model_dir}/scaler.joblib')
        self.label_encoder = joblib.load(f'{model_dir}/label_encoder.joblib')
        
        print(f"Model loaded from {model_dir}/")
    
    def predict_crop(self, features):
        """Predict crop based on input features"""
        # features should be a dict with keys: N, P, K, temperature, humidity, ph, rainfall
        feature_array = np.array([[
            features['N'], features['P'], features['K'], 
            features['temperature'], features['humidity'], 
            features['ph'], features['rainfall']
        ]])
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction = self.model.predict(feature_scaled)
        prediction_proba = self.model.predict_proba(feature_scaled)
        
        # Get crop name and confidence
        crop_name = self.label_encoder.inverse_transform(prediction)[0]
        confidence = max(prediction_proba[0]) * 100
        
        # Get top 3 recommendations
        top_indices = np.argsort(prediction_proba[0])[::-1][:3]
        recommendations = []
        
        for idx in top_indices:
            crop = self.label_encoder.inverse_transform([idx])[0]
            conf = prediction_proba[0][idx] * 100
            recommendations.append({'crop': crop, 'confidence': conf})
        
        return {
            'recommended_crop': crop_name,
            'confidence': confidence,
            'all_recommendations': recommendations
        }

def main():
    # Initialize model
    crop_model = CropRecommendationModel()
    
    # Train model
    accuracy = crop_model.train_model()
    
    # Save model
    crop_model.save_model()
    
    # Test prediction
    sample_input = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.8,
        'humidity': 82.0,
        'ph': 6.5,
        'rainfall': 202.9
    }
    
    result = crop_model.predict_crop(sample_input)
    print(f"\nSample Prediction:")
    print(f"Input: {sample_input}")
    print(f"Recommended Crop: {result['recommended_crop']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"All Recommendations: {result['all_recommendations']}")

if __name__ == "__main__":
    main()