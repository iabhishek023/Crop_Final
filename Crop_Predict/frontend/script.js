const API_BASE_URL = 'http://localhost:5000';
        
// Sample data for testing
const SAMPLE_DATA = {
    N: 90,
    P: 42,
    K: 43,
    temperature: 20.8,
    humidity: 82.0,
    ph: 6.5,
    rainfall: 202.9
};

function loadSampleData() {
    Object.keys(SAMPLE_DATA).forEach(key => {
        const input = document.querySelector(`input[name="${key}"]`);
        if (input) {
            input.value = SAMPLE_DATA[key];
        }
    });
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    const predictBtn = document.getElementById('predictBtn');
    
    if (show) {
        loading.style.display = 'block';
        predictBtn.disabled = true;
        predictBtn.textContent = 'Analyzing...';
    } else {
        loading.style.display = 'none';
        predictBtn.disabled = false;
        predictBtn.textContent = '🔮 Get Crop Recommendation';
    }
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    // Hide error after 5 seconds
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

function displayResults(result) {
    const resultSection = document.getElementById('resultSection');
    const recommendedCrop = document.getElementById('recommendedCrop');
    const confidence = document.getElementById('confidence');
    const allRecommendations = document.getElementById('allRecommendations');
    
    // Display main recommendation
    recommendedCrop.textContent = `🌱 ${result.recommended_crop.charAt(0).toUpperCase() + result.recommended_crop.slice(1)}`;
    confidence.textContent = `${result.confidence}%`;
    
    // Display all recommendations
    allRecommendations.innerHTML = '';
    result.all_recommendations.forEach((rec, index) => {
        const div = document.createElement('div');
        div.className = 'recommendation-item';
        div.innerHTML = `
            <span class="crop-name">${rec.crop}</span>
            <span class="crop-confidence">${rec.confidence}%</span>
        `;
        allRecommendations.appendChild(div);
    });
    
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

async function predictCrop(formData) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to get prediction');
        }

        const data = await response.json();
        return data;
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Cannot connect to server. Make sure the backend is running on http://localhost:5000');
        }
        throw error;
    }
}

document.getElementById('cropForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    hideError();
    showLoading(true);
    
    try {
        // Get form data
        const formData = new FormData(e.target);
        const data = {};
        
        // Convert form data to object with proper types
        for (let [key, value] of formData.entries()) {
            data[key] = parseFloat(value);
        }
        
        // Validate data
        for (let [key, value] of Object.entries(data)) {
            if (isNaN(value)) {
                throw new Error(`${key} must be a valid number`);
            }
        }
        
        // Make prediction
        const result = await predictCrop(data);
        
        if (result.success) {
            displayResults(result.result);
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
        
    } catch (error) {
        showError(error.message);
        document.getElementById('resultSection').style.display = 'none';
    } finally {
        showLoading(false);
    }
});

// Check backend health on page load
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('Backend is running and healthy');
        } else {
            showError('Backend is not responding properly');
        }
    } catch (error) {
        showError('Cannot connect to backend. Please start the Flask server.');
    }
}

// Initialize
window.addEventListener('load', () => {
    checkBackendHealth();
});