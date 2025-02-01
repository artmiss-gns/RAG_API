#!/bin/bash

# Start the cleanup script in the background, using nohup to avoid termination
nohup ./scripts/cleanup.sh >> /app/scripts/cleanup.log 2>&1 &

# Start the FastAPI application with uvicorn
exec uvicorn src.main:app --host 0.0.0.0 --port 8000
