#!/bin/bash
set -e

echo "🐍 Python version:"
python --version

echo "📦 Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "📥 Installing requirements..."
pip install --no-cache-dir -r requirements.txt

echo "✅ Build complete!"
