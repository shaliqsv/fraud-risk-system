import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

files_to_upload = [
    "train_transaction.csv",
    "train_identity.csv",
]

credential = DefaultAzureCredential()
storage_account = f"https://{os.getenv('AZURE_STORAGE_ACCOUNT')}.blob.core.windows.net"

# Added longer timeout — large files need more time
client = BlobServiceClient(
    account_url=storage_account,
    credential=credential,
    connection_timeout=600,
    read_timeout=600,
)

container_name = "fraud-risk-data"
try:
    client.create_container(container_name)
    print(f"Container '{container_name}' created")
except Exception:
    print(f"Container '{container_name}' already exists")

for filename in files_to_upload:
    filepath = f"data/raw/{filename}"
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Uploading {filename} ({file_size:.1f} MB)...")

    with open(filepath, "rb") as f:
        client.get_blob_client(container=container_name, blob=filename).upload_blob(
            f, overwrite=True, max_concurrency=4
        )

    print(f"✓ {filename} uploaded successfully")

print("\nAll files uploaded to Azure Blob Storage!")
