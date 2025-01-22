FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/app/venv

# Install required dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Create and activate virtual environment
#RUN uv venv -p 3.11 ${VENV_PATH}

# Activate virtual environment and install the package in editable mode
COPY . /app
#RUN . ${VENV_PATH}/bin/activate
RUN uv pip install --system -e .
#RUN uv pip install -e .

RUN uv pip install --system docling

EXPOSE 7860

CMD uv run --env-file=langflow.env langflow run
# Set entrypoint to use the virtual environment by default
#ENTRYPOINT ["/bin/bash", "-c", ". /app/venv/bin/activate && exec \"$@\"", "uv run --env-file=langflow.env langflow run"]
