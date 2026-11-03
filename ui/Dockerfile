FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 API_URL=http://credit-api
WORKDIR /app
COPY requirements/ui.txt requirements.txt
RUN pip install -r requirements.txt
COPY ui/app.py .
RUN useradd --create-home --uid 1000 appuser
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
