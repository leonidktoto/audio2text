pip install -U openai-whisper
docker build -t audio2text .  
docker run --rm -v ~/downloads/docker/asr:/app/ASR audio2text python /app/whisper_script.py /app/ASR large-v3