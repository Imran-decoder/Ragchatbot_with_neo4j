# Use Python 3.9 image
FROM python:3.10

# Set working directory
WORKDIR /code

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your application code
COPY . /code

# Create a non-root user (Security best practice for HF Spaces)
# This is required by Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the port 7860 (Hugging Face's default port)
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]