#!/bin/bash


nohup uvicorn model_api:app --proxy-headers --host "0.0.0.0" --port 5001

nohup streamlit run web_app.py