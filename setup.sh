#!/bin/bash

# setup.sh - Installation script for Streamlit deployment

echo "Starting package installation..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install core packages individually to identify any failures
pip install streamlit==1.28.2
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install scikit-learn==1.2.2
pip install joblib==1.2.0
pip install altair==4.2.2

echo "Package installation completed!"
