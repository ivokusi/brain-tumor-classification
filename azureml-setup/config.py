from azureml.core import Workspace
from dotenv import load_dotenv
import sys
import os

load_dotenv()

AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")

workspace_name = sys.argv[1]
resource_group = sys.argv[2]
location = sys.argv[3]

ws = Workspace.create(name=workspace_name,
                      subscription_id=AZURE_SUBSCRIPTION_ID,
                      resource_group=resource_group,
                      location=location)

ws.write_config(path=".azureml")
