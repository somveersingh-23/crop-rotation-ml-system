"""
Utility helper functions
"""
import re
from datetime import datetime
from typing import Optional, Tuple

def clean_crop_name(crop_name: str) -> str:
    """
    Clean crop name by removing Hindi text in parentheses
    
    Args:
        crop_name: Crop name like "Rice (धान)"
    
    Returns:
        Cleaned name: "Rice"
    """
    # Remove text in parentheses
    cleaned = re.sub(r'\s*\([^)]*\)', '', crop_name)
    return cleaned.strip()

def calculate_season_from_date(date: Optional[datetime] = None) -> str:
    """
    Calculate Indian agricultural season from date
    
    Args:
        date: Date to check (defaults to current date)
    
    Returns:
        Season name: "Kharif", "Rabi", or "Zaid"
    """
    if date is None:
        date = datetime.now()
    
    month = date.month
    
    # Kharif: June to October (monsoon season)
    if 6 <= month <= 10:
        return "Kharif"
    
    # Rabi: November to March (winter season)
    elif 11 <= month or month <= 3:
        return "Rabi"
    
    # Zaid: April to May (summer season)
    else:
        return "Zaid"

def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Validate geographic coordinates
    
    Args:
        latitude: Latitude value
        longitude: Longitude value
    
    Returns:
        True if valid, False otherwise
    """
    if not (-90 <= latitude <= 90):
        return False
    
    if not (-180 <= longitude <= 180):
        return False
    
    return True

def format_crop_duration(days: int) -> str:
    """
    Format crop duration in human-readable format
    
    Args:
        days: Number of days
    
    Returns:
        Formatted string like "90-120 days" or "12-18 months"
    """
    if days < 365:
        # Format in days
        lower = days - 10
        upper = days + 10
        return f"{lower}-{upper} days"
    else:
        # Format in months
        months = days // 30
        lower = months - 1
        upper = months + 1
        return f"{lower}-{upper} months"

def calculate_confidence_level(confidence: float) -> str:
    """
    Convert confidence score to human-readable level
    
    Args:
        confidence: Confidence score (0-1)
    
    Returns:
        Confidence level: "High", "Medium", "Low"
    """
    if confidence >= 0.8:
        return "High"
    elif confidence >= 0.6:
        return "Medium"
    else:
        return "Low"

def get_npk_recommendation(crop: str) -> Tuple[int, int, int]:
    """
    Get NPK fertilizer recommendation for crop
    
    Args:
        crop: Crop name
    
    Returns:
        Tuple of (N, P, K) values
    """
    recommendations = {
        "Rice": (80, 40, 40),
        "Wheat": (120, 60, 40),
        "Maize": (100, 50, 40),
        "Cotton": (100, 50, 50),
        "Soybean": (20, 60, 40),
        "Groundnut": (20, 60, 60),
        "Mustard": (80, 40, 20),
        "Potato": (150, 80, 80),
        "Chickpea": (20, 60, 30),
        "Sugarcane": (250, 100, 100)
    }
    
    crop_clean = clean_crop_name(crop)
    
    for crop_key, npk in recommendations.items():
        if crop_key in crop_clean:
            return npk
    
    # Default recommendation
    return (100, 50, 50)

def format_price(price: float, currency: str = "₹") -> str:
    """
    Format price in Indian currency
    
    Args:
        price: Price value
        currency: Currency symbol
    
    Returns:
        Formatted price string
    """
    if price >= 10000000:  # 1 crore
        crores = price / 10000000
        return f"{currency}{crores:.2f} Cr"
    elif price >= 100000:  # 1 lakh
        lakhs = price / 100000
        return f"{currency}{lakhs:.2f} L"
    elif price >= 1000:  # 1 thousand
        thousands = price / 1000
        return f"{currency}{thousands:.2f} K"
    else:
        return f"{currency}{price:.2f}"

def calculate_yield_value(yield_str: str, price_per_quintal: float) -> Optional[float]:
    """
    Calculate total value from yield and price
    
    Args:
        yield_str: Yield string like "4-5 tons/ha"
        price_per_quintal: Price per quintal
    
    Returns:
        Estimated value in rupees
    """
    try:
        # Extract numbers from yield string
        numbers = re.findall(r'\d+', yield_str)
        if len(numbers) >= 2:
            avg_yield_tons = (int(numbers[0]) + int(numbers[1])) / 2
            avg_yield_quintals = avg_yield_tons * 10  # 1 ton = 10 quintals
            return avg_yield_quintals * price_per_quintal
    except:
        pass
    
    return None

def get_soil_type_description(soil_type: str) -> str:
    """
    Get detailed description of soil type
    
    Args:
        soil_type: Soil type name
    
    Returns:
        Description string
    """
    descriptions = {
        "Alluvial": "Highly fertile soil formed by river deposits, rich in nutrients",
        "Black": "Deep black soil with high clay content, excellent water retention",
        "Red": "Red colored soil rich in iron, good drainage",
        "Laterite": "Acidic soil formed in high rainfall areas, low fertility",
        "Arid": "Dry sandy soil with low organic content",
        "Mountain": "Soil found in hilly areas, varies in composition"
    }
    
    soil_clean = clean_crop_name(soil_type)
    
    for soil_key, desc in descriptions.items():
        if soil_key.lower() in soil_clean.lower():
            return desc
    
    return "Agricultural soil suitable for cultivation"
