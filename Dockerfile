FROM python:3.12.3

WORKDIR /app

RUN export PYTHONPATH="/$(pwd):$PYTHONPATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# RUN python3 ./src/nltk_fix.py

# EXPOSE 8000
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
