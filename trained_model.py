# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workspace
client = InferenceHTTPClient(
  api_url="https://serverless.roboflow.com",
  api_key="NrTZJtIhHLeGN1e8pEO4"
)

# 3. Run your workflow on an image
result = client.run_workflow(
  workspace_name="objectvisualization",
  workflow_id="general-segmentation-api-2",
  images={
    "image": "test.jpg"  # Path to your image file
  },
  parameters={
    "classes": "fire"
  },
  use_cache=True  # cache workflow definition for 15 minutes
)

# 4. Get your results
print(result)