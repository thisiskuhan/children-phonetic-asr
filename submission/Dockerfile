# Minimal local testing (CPU). For GPU testing, use the competition runtime:
#   cd childrens-speech-recognition-runtime && just track=phonetic run
#
# Build:
#   docker build -t pasketti-test -f submission/Dockerfile .
#
# Run (from project root):
#   docker run --rm \
#     -v $(pwd)/submission/src:/code_execution/src:ro \
#     -v $(pwd)/childrens-speech-recognition-runtime/data-demo/phonetic:/code_execution/data:ro \
#     -v $(pwd)/submission/output:/code_execution/submission \
#     -e DECODER=greedy \
#     pasketti-test

FROM python:3.11-slim

RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers

WORKDIR /code_execution
RUN mkdir -p submission

CMD ["python", "src/main.py"]
