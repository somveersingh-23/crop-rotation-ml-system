"""
Crop rotation rules and agricultural knowledge base
"""
from typing import List, Dict, Set

class CropRotationRules:
    """Agricultural knowledge base for crop rotation"""
    
    def __init__(self):
        # Crop families
        self.crop_families = {
            "Legumes": ["Soybean", "Groundnut", "Chickpea", "Lentil"],
            "Cereals": ["Rice", "Wheat", "Maize", "Bajra", "Jowar", "Barley"],
            "Oilseeds": ["Mustard", "Sunflower"],
            "Fiber": ["Cotton"],
            "Tubers": ["Potato"],
            "Cash_Crops": ["Sugarcane"]
        }
        
        # Nitrogen fixers
        self.nitrogen_fixers = ["Soybean", "Groundnut", "Chickpea", "Lentil"]
        
        # Heavy vs light feeders
        self.heavy_feeders = ["Rice", "Wheat", "Maize", "Cotton", "Potato", "Sugarcane"]
        self.light_feeders = ["Mustard", "Bajra", "Jowar", "Pulses"]
        
        # Root depth
        self.deep_rooted = ["Cotton", "Sugarcane", "Sunflower"]
        self.shallow_rooted = ["Wheat", "Rice", "Potato"]
        
        # Season mapping
        self.season_crops = {
            "Kharif": ["Rice", "Maize", "Cotton", "Soybean", "Groundnut", "Bajra", "Jowar"],
            "Rabi": ["Wheat", "Mustard", "Potato", "Chickpea", "Barley", "Lentil"],
            "Zaid": ["Vegetables", "Cucumber", "Watermelon"],
            "Year-round": ["Sugarcane"]
        }
        
        # Soil preferences
        self.soil_preferences = {
            "alluvial": ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton"],
            "black": ["Cotton", "Soybean", "Wheat", "Chickpea", "Sunflower"],
            "red": ["Groundnut", "Cotton", "Rice", "Maize"],
            "laterite": ["Rice"],
            "arid": ["Bajra", "Jowar", "Mustard"],
            "mountain": ["Wheat", "Barley", "Potato"]
        }
        
        # Crop icons (Unicode emoji)
        self.crop_icons = {
            "Rice": "🌾", "Wheat": "🌾", "Maize": "🌽", "Cotton": "🤍",
            "Soybean": "🫘", "Groundnut": "🥜", "Mustard": "🌻", "Bajra": "🌾",
            "Potato": "🥔", "Chickpea": "🫘", "Sugarcane": "🎋", "Sunflower": "🌻",
            "Jowar": "🌾", "Barley": "🌾", "Lentil": "🫘"
        }
        
        # Crop durations (in days)
        self.crop_durations = {
            "Rice": "120-150 days",
            "Wheat": "120-140 days",
            "Maize": "90-110 days",
            "Cotton": "180-200 days",
            "Soybean": "90-120 days",
            "Groundnut": "100-130 days",
            "Mustard": "100-120 days",
            "Bajra": "70-90 days",
            "Potato": "90-120 days",
            "Chickpea": "120-140 days",
            "Sugarcane": "12-18 months",
            "Sunflower": "90-110 days",
            "Jowar": "100-120 days",
            "Barley": "120-140 days",
            "Lentil": "120-140 days"
        }
        
        # Expected yields (per hectare)
        self.expected_yields = {
            "Rice": "4-5 tons/ha",
            "Wheat": "3-4 tons/ha",
            "Maize": "5-6 tons/ha",
            "Cotton": "2-3 tons/ha",
            "Soybean": "2-3 tons/ha",
            "Groundnut": "2-3 tons/ha",
            "Mustard": "1-2 tons/ha",
            "Bajra": "2-3 tons/ha",
            "Potato": "25-30 tons/ha",
            "Chickpea": "2-3 tons/ha",
            "Sugarcane": "70-80 tons/ha",
            "Sunflower": "1.5-2 tons/ha",
            "Jowar": "2-3 tons/ha",
            "Barley": "2.5-3 tons/ha",
            "Lentil": "1.5-2 tons/ha"
        }
    
    def get_crop_family(self, crop: str) -> str:
        """Get family of a crop"""
        crop_clean = crop.split("(")[0].strip()
        for family, crops in self.crop_families.items():
            if any(c in crop_clean for c in crops):
                return family
        return "Unknown"
    
    def should_avoid_sequence(self, crop1: str, crop2: str) -> bool:
        """Check if two crops should not follow each other"""
        family1 = self.get_crop_family(crop1)
        family2 = self.get_crop_family(crop2)
        
        # Avoid same family
        if family1 == family2 and family1 != "Unknown":
            return True
        
        # Avoid two heavy feeders
        crop1_clean = crop1.split("(")[0].strip()
        crop2_clean = crop2.split("(")[0].strip()
        
        if any(c in crop1_clean for c in self.heavy_feeders) and \
           any(c in crop2_clean for c in self.heavy_feeders):
            return True
        
        return False
    
    def get_rotation_score(self, current: str, previous: List[str], next_crop: str) -> float:
        """Calculate rotation compatibility score (0-1)"""
        score = 1.0
        
        # Penalize same family
        if previous and self.should_avoid_sequence(previous[-1], next_crop):
            score -= 0.3
        
        # Penalize if same as current
        if current.split("(")[0].strip() == next_crop.split("(")[0].strip():
            score -= 0.5
        
        # Bonus for nitrogen fixer after heavy feeder
        current_clean = current.split("(")[0].strip()
        next_clean = next_crop.split("(")[0].strip()
        
        if any(c in current_clean for c in self.heavy_feeders) and \
           any(c in next_clean for c in self.nitrogen_fixers):
            score += 0.3
        
        # Bonus for root depth alternation
        if any(c in current_clean for c in self.deep_rooted) and \
           any(c in next_clean for c in self.shallow_rooted):
            score += 0.2
        elif any(c in current_clean for c in self.shallow_rooted) and \
             any(c in next_clean for c in self.deep_rooted):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def get_compatible_crops(self, soil_type: str, season: str) -> List[str]:
        """Get crops compatible with soil and season"""
        soil_key = soil_type.lower().replace("/", "_").replace(" ", "_")
        
        # Map to standard soil types
        soil_map = {
            "alluvial": "alluvial", "black": "black", "regur": "black",
            "red": "red", "laterite": "laterite", "arid": "arid",
            "desert": "arid", "mountain": "mountain"
        }
        
        for key, value in soil_map.items():
            if key in soil_key:
                soil_key = value
                break
        
        soil_crops = set(self.soil_preferences.get(soil_key, []))
        season_crops = set(self.season_crops.get(season, []))
        
        compatible = list(soil_crops.intersection(season_crops)) if season_crops else list(soil_crops)
        return compatible if compatible else list(soil_crops)
    
    def get_crop_icon(self, crop: str) -> str:
        """Get emoji icon for crop"""
        crop_clean = crop.split("(")[0].strip()
        for crop_name, icon in self.crop_icons.items():
            if crop_name in crop_clean:
                return icon
        return "🌾"
    
    def get_crop_duration(self, crop: str) -> str:
        """Get typical duration for crop"""
        crop_clean = crop.split("(")[0].strip()
        for crop_name, duration in self.crop_durations.items():
            if crop_name in crop_clean:
                return duration
        return "90-120 days"
    
    def get_expected_yield(self, crop: str) -> str:
        """Get expected yield for crop"""
        crop_clean = crop.split("(")[0].strip()
        for crop_name, yield_val in self.expected_yields.items():
            if crop_name in crop_clean:
                return yield_val
        return "2-3 tons/ha"

# Global instance
crop_rules = CropRotationRules()
