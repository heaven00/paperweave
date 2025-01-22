# paperweave
Interactive notebooklm


# Project Setup

if you are using another way to setup the environment that is also okay, you can document it here
### Setup Environment using UV
- On macOS and Linux.
```curl -LsSf https://astral.sh/uv/install.sh | sh```
- On windows
```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

copied from https://github.com/astral-sh/uv


- create virtualenv with python 3.11 `uv -p 3.11 venv`
- activate virtualenv `source venv/bin/activate`
- install requirements `uv pip install -r requirements.txt`

### Running langflow
*make sure the environment is activated*

`DO_NOT_TRACK=true LANGFLOW_COMPONENTS_PATH=src/components langflow run`

- do not track to remove tracking 
- langflow components path to load in custom components we might create for our use case

### versioning our flows
- Download the flow as json using the export feature and put them under `src/flows` directory
- commit to github
- To load anyone's else flow use the import flow feature

