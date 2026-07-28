FROM python:3.11-slim-bookworm

# Offline runtime: tesseract plus the English traineddata, no model downloads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# RapidOCR carries the PP-OCRv4 mobile ONNX models inside the wheel (16 MB
# total), so nothing is fetched at runtime. Tesseract stays as the fallback
# engine and keeps the A/B switch in mib/ocr.py reproducible.
RUN pip install --no-cache-dir \
        PyMuPDF==1.24.10 \
        opencv-python-headless==4.10.0.84 \
        pytesseract==0.3.13 \
        rapidocr-onnxruntime==1.4.4 \
        onnxruntime==1.20.1 \
        numpy==1.26.4

WORKDIR /app
COPY mib/ /app/mib/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# The scoring host mounts the root filesystem read-only.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_THREAD_LIMIT=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

ENTRYPOINT ["/app/run.sh"]
