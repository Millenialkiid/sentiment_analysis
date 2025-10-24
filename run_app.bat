@echo off
title Sentiment Analysis - App
echo =====================================================
echo  Launching Streamlit App
echo =====================================================

REM Activate virtual environment
call .venv\Scripts\activate

echo Starting the app...
streamlit run app.py

pause
