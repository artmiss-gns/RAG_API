FROM python:3.12.3

WORKDIR /app

RUN export PYTHONPATH="/$(pwd):$PYTHONPATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Make scripts executable
RUN chmod +x scripts/cleanup.sh
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

# Use the entrypoint script to run both processes
CMD ["./scripts/entrypoint.sh"]