FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download glasspy model files from Zenodo at build time (network available).
# At runtime the versions.json cache will exist, so glasspy skips the download.
RUN python -c "import glasspy"

# Copy app code
COPY . .

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app_glass.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
