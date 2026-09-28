@echo off
title AI Smart Vision & Health Demo - Assignment 5
cd /d "%~dp0"
echo ========================================================
echo   KHOI DONG WEB DEMO AI SMART SYSTEM (ASSIGNMENT 5)
echo ========================================================
echo Web se tu dong mo tren trinh duyet tai: http://localhost:8501
echo Nhan Ctrl + C trong cua so nay de dung ung dung.
echo.
"C:\Users\agung\anaconda3\python.exe" -m streamlit run app.py
pause
