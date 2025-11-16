"""
API endpoint tests using pytest
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.api.schemas import CropRotationRequest, Location, Weather

# Create test client
client = TestClient(app)

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_loaded" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data

class TestCropRotationPrediction:
    """Test crop rotation prediction endpoint"""
    
    def test_basic_prediction(self):
        """Test basic crop rotation prediction"""
        request_data = {
            "currentCrop": "Rice (धान)",
            "previousCrops": ["Wheat (गेहूं)"],
            "soilType": "Alluvial",
            "weather": {
                "temperature": 28,
                "humidity": 70,
                "rainfall": 850,
                "season": "Kharif"
            },
            "includeMarketTrends": True,
            "rotationDuration": 2
        }
        
        response = client.post("/api/crop-rotation/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "rotationPlan" in data
        assert len(data["rotationPlan"]) > 0
        assert "soilHealthImprovement" in data
        assert "estimatedProfitIncrease" in data
    
    def test_prediction_with_location(self):
        """Test prediction with location coordinates"""
        request_data = {
            "currentCrop": "Wheat (गेहूं)",
            "previousCrops": [],
            "soilType": "Black",
            "location": {
                "latitude": 28.6139,
                "longitude": 77.2090,
                "name": "Delhi"
            },
            "includeMarketTrends": False,
            "rotationDuration": 2
        }
        
        response = client.post("/api/crop-rotation/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["currentCrop"] == "Wheat (गेहूं)"
    
    def test_prediction_multiple_previous_crops(self):
        """Test with multiple previous crops"""
        request_data = {
            "currentCrop": "Cotton (कपास)",
            "previousCrops": ["Rice (धान)", "Wheat (गेहूं)", "Soybean (सोयाबीन)"],
            "soilType": "Red",
            "weather": {
                "temperature": 27,
                "humidity": 55,
                "rainfall": 700,
                "season": "Kharif"
            },
            "includeMarketTrends": True,
            "rotationDuration": 3
        }
        
        response = client.post("/api/crop-rotation/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["rotationPlan"]) > 0
        
        # Verify each crop in plan has required fields
        for crop in data["rotationPlan"]:
            assert "cropName" in crop
            assert "season" in crop
            assert "benefits" in crop
            assert "expectedYield" in crop
            assert "sequenceNumber" in crop
    
    def test_invalid_request_missing_fields(self):
        """Test validation with missing required fields"""
        request_data = {
            "currentCrop": "Rice (धान)"
            # Missing soilType
        }
        
        response = client.post("/api/crop-rotation/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_temperature(self):
        """Test validation with invalid temperature"""
        request_data = {
            "currentCrop": "Rice (धान)",
            "soilType": "Alluvial",
            "weather": {
                "temperature": 100,  # Invalid temp
                "humidity": 70,
                "rainfall": 850,
                "season": "Kharif"
            },
            "rotationDuration": 2
        }
        
        response = client.post("/api/crop-rotation/predict", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_season(self):
        """Test validation with invalid season"""
        request_data = {
            "currentCrop": "Rice (धान)",
            "soilType": "Alluvial",
            "weather": {
                "temperature": 28,
                "humidity": 70,
                "rainfall": 850,
                "season": "InvalidSeason"  # Invalid
            },
            "rotationDuration": 2
        }
        
        response = client.post("/api/crop-rotation/predict", json=request_data)
        assert response.status_code == 422
    
    def test_rotation_duration_limits(self):
        """Test rotation duration edge cases"""
        # Test minimum duration
        request_data = {
            "currentCrop": "Rice (धान)",
            "soilType": "Alluvial",
            "weather": {
                "temperature": 28,
                "humidity": 70,
                "rainfall": 850,
                "season": "Kharif"
            },
            "rotationDuration": 1
        }
        
        response = client.post("/api/crop-rotation/predict", json=request_data)
        assert response.status_code == 200
        
        # Test maximum duration
        request_data["rotationDuration"] = 5
        response = client.post("/api/crop-rotation/predict", json=request_data)
        assert response.status_code == 200
    
    def test_response_structure(self):
        """Test complete response structure"""
        request_data = {
            "currentCrop": "Maize (मक्का)",
            "previousCrops": ["Wheat (गेहूं)"],
            "soilType": "Alluvial",
            "weather": {
                "temperature": 25,
                "humidity": 60,
                "rainfall": 800,
                "season": "Kharif"
            },
            "includeMarketTrends": True,
            "rotationDuration": 2
        }
        
        response = client.post("/api/crop-rotation/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify top-level structure
        assert isinstance(data["success"], bool)
        assert isinstance(data["currentCrop"], str)
        assert isinstance(data["rotationPlan"], list)
        assert isinstance(data["overallBenefits"], list)
        assert isinstance(data["soilHealthImprovement"], float)
        assert isinstance(data["estimatedProfitIncrease"], float)
        assert isinstance(data["confidenceScore"], float)
        
        # Verify rotation plan structure
        if len(data["rotationPlan"]) > 0:
            crop = data["rotationPlan"][0]
            assert "cropName" in crop
            assert "season" in crop
            assert "duration" in crop
            assert "benefits" in crop
            assert "expectedYield" in crop
            assert "sequenceNumber" in crop
            assert "icon" in crop

class TestModelIntegration:
    """Test ML model integration"""
    
    def test_model_predictions_consistency(self):
        """Test that same input gives consistent results"""
        request_data = {
            "currentCrop": "Rice (धान)",
            "soilType": "Alluvial",
            "weather": {
                "temperature": 28,
                "humidity": 70,
                "rainfall": 850,
                "season": "Kharif"
            },
            "rotationDuration": 2
        }
        
        # Make two identical requests
        response1 = client.post("/api/crop-rotation/predict", json=request_data)
        response2 = client.post("/api/crop-rotation/predict", json=request_data)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Results should be identical
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["rotationPlan"] == data2["rotationPlan"]

# Pytest configuration
@pytest.fixture
def sample_request():
    """Sample request data fixture"""
    return {
        "currentCrop": "Rice (धान)",
        "previousCrops": ["Wheat (गेहूं)"],
        "soilType": "Alluvial",
        "weather": {
            "temperature": 28,
            "humidity": 70,
            "rainfall": 850,
            "season": "Kharif"
        },
        "includeMarketTrends": True,
        "rotationDuration": 2
    }

def test_sample_request_fixture(sample_request):
    """Test using fixture"""
    response = client.post("/api/crop-rotation/predict", json=sample_request)
    assert response.status_code == 200

# Run tests with: pytest tests/test_api.py -v
