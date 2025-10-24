@echo off
title Sentiment Analysis - Training
echo =====================================================
echo  Activating virtual environment and training model
echo =====================================================

REM Activate virtual environment
call .venv\Scripts\activate

echo Running training script...
python scripts/train.py

echo =====================================================
echo  Training complete! Check logs or models/ directory
echo =====================================================

pause
