# Use the official Python image as a base
FROM python:3.11.9-slim


RUN apt-get update && apt-get install -y ffmpeg && apt-get clean
# Set the working directory inside the container
WORKDIR /app

# Install the dependencies
RUN pip install -U openai-whisper

RUN python -c "import whisper; whisper.load_model('base'); whisper.load_model('small');"

# Copy the rest of the application code to the container
COPY . .

# Set the command to run your script
CMD ["python", "whisper_script.py"]