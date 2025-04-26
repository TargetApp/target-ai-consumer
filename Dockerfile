FROM python:3.12.10-slim

# Set the working directory
WORKDIR /usr/src/target-ai-consumer

# Copy the source code
COPY core /usr/src/target-ai-consumer/core
COPY interfaces /usr/src/target-ai-consumer/interfaces
COPY models /usr/src/target-ai-consumer/models
COPY *.py /usr/src/target-ai-consumer/
COPY pyproject.toml /usr/src/target-ai-consumer/pyproject.toml

# Install the dependencies
RUN apt update
RUN apt install -y libgl1-mesa-glx libglib2.0-0 python3-poetry
RUN pip install --upgrade poetry
RUN poetry install --no-root --without dev

# Run the application
ENTRYPOINT ["poetry", "run", "python", "main.py", "settings.toml"]
CMD ["both"]