import os

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


def get_service_credential(key_env_var: str | None = None):
    if key_env_var:
        api_key = os.getenv(key_env_var)
        if api_key:
            return AzureKeyCredential(api_key)
    return DefaultAzureCredential()


def get_blob_service_client() -> BlobServiceClient:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    if not account_name:
        raise ValueError(
            "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME before using Azure Blob Storage."
        )

    return BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )