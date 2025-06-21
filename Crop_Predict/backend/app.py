from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

class CropPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            self.model = joblib.load(f'{self.model_dir}/crop_model.joblib')
            self.scaler = joblib.load(f'{self.model_dir}/scaler.joblib')
            self.label_encoder = joblib.load(f'{self.model_dir}/label_encoder.joblib')
            logger.info("Model loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise Exception("Model not found. Please train the model first.")
    
    def validate_input(self, features):
        """Validate input features"""
        required_features = self.feature_names
        
        # Check if all required features are present
        for feature in required_features:
            if feature not in features:
                return False, f"Missing feature: {feature}"
        
        # Validate feature ranges
        validations = {
            'N': (0, 500),
            'P': (0, 200),
            'K': (0, 500),
            'temperature': (-10, 50),
            'humidity': (0, 100),
            'ph': (0, 14),
            'rainfall': (0, 500)
        }
        
        for feature, (min_val, max_val) in validations.items():
            value = features[feature]
            if not isinstance(value, (int, float)):
                return False, f"{feature} must be a number"
            if not (min_val <= value <= max_val):
                return False, f"{feature} must be between {min_val} and {max_val}"
        
        return True, "Valid input"
    
    def predict(self, features):
        """Predict crop based on input features"""
        # Validate input
        is_valid, message = self.validate_input(features)
        if not is_valid:
            raise ValueError(message)
        
        # Prepare feature array
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
            recommendations.append({
                'crop': crop, 
                'confidence': round(conf, 2),
                'probability': round(conf/100, 4)
            })
        
        return {
            'recommended_crop': crop_name,
            'confidence': round(confidence, 2),
            'all_recommendations': recommendations,
            'input_features': features,
            'timestamp': datetime.now().isoformat()
        }

# Initialize predictor
try:
    predictor = CropPredictor()
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route('/')
def home():
    """Home page"""
    return jsonify({
        "message": "🌾 Crop Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Get crop recommendation",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information",
            "/sample-input": "GET - Sample input format"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if predictor and predictor.model else "not loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/model-info')
def model_info():
    """Get model information"""
    if not predictor or not predictor.model:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": "Random Forest Classifier",
        "features": predictor.feature_names,
        "crops": predictor.label_encoder.classes_.tolist(),
        "n_estimators": predictor.model.n_estimators,
        "feature_importance": dict(zip(
            predictor.feature_names, 
            predictor.model.feature_importances_.tolist()
        ))
    })

@app.route('/sample-input')
def sample_input():
    """Get sample input format"""
    return jsonify({
        "sample_input": {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 20.8,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.9
        },
        "feature_descriptions": {
            "N": "Nitrogen content in soil (0-500)",
            "P": "Phosphorous content in soil (0-200)",
            "K": "Potassium content in soil (0-500)",
            "temperature": "Temperature in degree Celsius (-10 to 50)",
            "humidity": "Relative humidity in % (0-100)",
            "ph": "pH value of the soil (0-14)",
            "rainfall": "Rainfall in mm (0-500)"
        }
    })

@app.route('/predict', methods=['POST'])
def predict_crop():
    """Predict crop based on input features"""
    if not predictor or not predictor.model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Make prediction
        result = predictor.predict(data)
        
        # Log the prediction
        logger.info(f"Prediction made: {result['recommended_crop']} with {result['confidence']}% confidence")
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Predict crops for multiple inputs"""
    if not predictor or not predictor.model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'inputs' not in data:
            return jsonify({"error": "No inputs provided"}), 400
        
        inputs = data['inputs']
        if not isinstance(inputs, list):
            return jsonify({"error": "Inputs must be a list"}), 400
        
        results = []
        for i, input_data in enumerate(inputs):
            try:
                result = predictor.predict(input_data)
                results.append({
                    "index": i,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "results": results,
            "total_predictions": len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if model exists, if not, provide instructions
    if not os.path.exists('models/crop_model.joblib'):
        print("⚠️  Model not found!")
        print("Please run the following command to train the model first:")
        print("python ml_model/crop_model.py")
        print("\nStarting server anyway for testing purposes...")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)