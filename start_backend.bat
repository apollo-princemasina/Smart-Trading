@echo off
cd /d "C:\Users\ndlov\Documents\Research and Innovation\Smart Trading"
python -m uvicorn src.api.main:app --reload --port 8000 > backend_startup.log 2>&1
