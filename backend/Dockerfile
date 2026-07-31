FROM python:3.10-slim

WORKDIR /app

# Install build dependencies for Cython
RUN apt-get update && apt-get install -y build-essential gcc g++ && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 support first (helps with caching)
# Using index-url to get the GPU version
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build Cython extension
WORKDIR /app/cython_modules
RUN python setup.py build_ext --inplace

WORKDIR /app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
