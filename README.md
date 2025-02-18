# paperweave
Interactive notebooklm


# Project Setup

if you are using another way to setup the environment that is also okay, you can document it here
### Setup Environment using UV
- On macOS and Linux.
```curl -LsSf https://astral.sh/uv/install.sh | sh```
Alternatively, you can use pip install to install uv
'''pip install uv'''
- On windows
```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

copied from https://github.com/astral-sh/uv


- Run `uv sync` to create a virtual environment for your project.This adds paperweave also to your python environment.
- Use `uv run python your_script.py` to run your python script within the virtual environment or activate the environment manually by running `source .venv/bin/activate`.
