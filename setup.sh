#!/bin/bash

echo "🚀 Setting up Crop Rotation ML System..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create directories
mkdir -p data/raw data/processed models logs notebooks

# Copy environment files
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please update .env file with your API keys"
fi

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle API setup required:"
    echo "   1. Go to https://www.kaggle.com/settings/account"
    echo "   2. Click 'Create New API Token'"
    echo "   3. Download kaggle.json"
    echo "   4. Move it to ~/.kaggle/kaggle.json"
    echo "   5. Run: chmod 600 ~/.kaggle/kaggle.json"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Configure Kaggle API: mv kaggle.json ~/.kaggle/"
echo "2. Update .env with OpenWeather API key"
echo "3. Run: python data/download_datasets.py"
echo "4. Run: python scripts/train_model.py"
