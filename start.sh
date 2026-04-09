#!/bin/bash
set -e

# Start FastAPI in background
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 &

# Wait until FastAPI is ready using Python
echo "Waiting for FastAPI to start..."
python3 - <<END
import socket, time
while True:
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=1):
            break
    except OSError:
        time.sleep(0.5)
END
echo "FastAPI is up!"

# Start Streamlit
streamlit run src/app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false