"""
Main crop rotation recommendation engine
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from app.ml.predictor import predictor
from app.ml.crop_rules import crop_rules
from app.config import settings

class CropRotationModel:
    """Main model for crop rotation recommendations"""
    
    def __init__(self):
        self.predictor = predictor
        self.rules = crop_rules
        self.market_data = self.load_market_data()
    
    def load_market_data(self) -> pd.DataFrame:
        """Load market price data"""
        try:
            return pd.read_csv(settings.MARKET_PRICES_PATH)
        except:
            return pd.DataFrame()
    
    def predict_rotation(
        self,
        current_crop: str,
        previous_crops: List[str],
        soil_type: str,
        weather: Dict,
        duration: int,
        include_market: bool
    ) -> List[Dict]:
        """Generate crop rotation plan"""
        
        recommendations = []
        season = weather.get("season", "Kharif")
        
        # Get compatible crops
        compatible_crops = self.rules.get_compatible_crops(soil_type, season)
        
        # Get ML predictions for best crops
        ml_predictions = self.predictor.predict_best_crops(
            N=100,  # Default NPK values (can be customized)
            P=50,
            K=50,
            temperature=weather.get("temperature", 25),
            humidity=weather.get("humidity", 65),
            pH=6.5,  # Default pH
            rainfall=weather.get("rainfall", 800),
            top_k=10
        )
        
        # Score and rank crops
        crop_scores = []
        for ml_pred in ml_predictions:
            crop = ml_pred['crop']
            
            # Skip if not compatible with soil/season
            if crop not in compatible_crops:
                continue
            
            # Skip current crop
            if crop.split("(")[0].strip() == current_crop.split("(")[0].strip():
                continue
            
            # Calculate rotation score
            rotation_score = self.rules.get_rotation_score(current_crop, previous_crops, crop)
            
            # Combined score: ML confidence (60%) + rotation score (40%)
            total_score = (ml_pred['confidence'] * 0.6) + (rotation_score * 0.4)
            
            crop_scores.append((crop, total_score, ml_pred['confidence']))
        
        # Sort by score
        crop_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Generate rotation plan
        sequence_num = 1
        used_crops = []
        seasons = self.get_season_sequence(season, duration * 2)
        
        for i, (crop, score, confidence) in enumerate(crop_scores[:duration * 2]):
            if crop in used_crops:
                continue
            
            current_season = seasons[i % len(seasons)]
            
            # Check season compatibility
            if not self.is_season_compatible(crop, current_season):
                continue
            
            recommendations.append({
                "cropName": crop,
                "season": current_season,
                "duration": self.rules.get_crop_duration(crop),
                "benefits": self.get_crop_benefits(crop, current_crop, previous_crops + used_crops),
                "expectedYield": self.rules.get_expected_yield(crop),
                "marketDemand": self.get_market_demand(crop, include_market),
                "profitability": self.get_profitability(crop, score, include_market),
                "sequenceNumber": sequence_num,
                "icon": self.rules.get_crop_icon(crop)
            })
            
            used_crops.append(crop)
            sequence_num += 1
            
            if len(recommendations) >= min(duration * 2, 4):  # Max 4 recommendations
                break
        
        return recommendations
    
    def get_crop_benefits(self, crop: str, current_crop: str, previous_crops: List[str]) -> List[str]:
        """Get specific benefits"""
        benefits = []
        crop_clean = crop.split("(")[0].strip()
        current_clean = current_crop.split("(")[0].strip()
        
        if any(leg in crop_clean for leg in self.rules.nitrogen_fixers):
            benefits.append("Fixes nitrogen naturally, enriches soil")
        
        if any(hf in current_clean for hf in self.rules.heavy_feeders):
            if crop_clean in self.rules.light_feeders or crop_clean in self.rules.nitrogen_fixers:
                benefits.append("Allows soil recovery after heavy feeding crop")
        
        if self.rules.get_crop_family(crop) != self.rules.get_crop_family(current_crop):
            benefits.append("Breaks pest and disease cycles")
        
        if not benefits:
            benefits = ["Diversifies farm income", "Improves soil health"]
        
        return benefits[:3]
    
    def get_market_demand(self, crop: str, include_market: bool) -> str:
        """Get market demand"""
        if not include_market or self.market_data.empty:
            return None
        
        crop_clean = crop.split("(")[0].strip()
        row = self.market_data[self.market_data['crop'].str.contains(crop_clean, case=False)]
        
        if not row.empty:
            return row.iloc[0]['demand_level']
        
        return "Moderate Demand"
    
    def get_profitability(self, crop: str, score: float, include_market: bool) -> str:
        """Calculate profitability"""
        if not include_market:
            return None
        
        if score > 0.8:
            return "High Profit"
        elif score > 0.6:
            return "Good Profit"
        else:
            return "Moderate Profit"
    
    def get_season_sequence(self, start_season: str, count: int) -> List[str]:
        """Generate season sequence"""
        seasons = ["Kharif", "Rabi", "Zaid"]
        try:
            start_idx = seasons.index(start_season)
        except:
            start_idx = 0
        
        return [seasons[(start_idx + i) % len(seasons)] for i in range(count)]
    
    def is_season_compatible(self, crop: str, season: str) -> bool:
        """Check season compatibility"""
        crop_clean = crop.split("(")[0].strip()
        season_crops = self.rules.season_crops.get(season, [])
        return any(c in crop_clean for c in season_crops)

model = CropRotationModel()
