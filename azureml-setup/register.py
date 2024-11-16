from azureml.core import Workspace, Model
from dotenv import load_dotenv
import sys
import os

load_dotenv()

AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")

model_name = sys.argv[1]
model_path = sys.argv[2]
description = f"Uploaded {model_name} model"

ws = Workspace.from_config()

model = Model.register(workspace=ws,
                       model_name=model_name,
                       model_path=model_path,
                       description=description)

print(f'Model registered: {model.name}, Version: {model.version}')
