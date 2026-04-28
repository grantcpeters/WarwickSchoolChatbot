import json
from typing import Any

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient


def load_json_state(blob_client: BlobServiceClient, container: str, blob_name: str) -> dict[str, Any]:
    container_client = blob_client.get_container_client(container)
    try:
        data = container_client.download_blob(blob_name).readall()
    except ResourceNotFoundError:
        return {}

    if not data:
        return {}

    return json.loads(data.decode("utf-8"))


def save_json_state(
    blob_client: BlobServiceClient,
    container: str,
    blob_name: str,
    payload: dict[str, Any],
) -> None:
    container_client = blob_client.get_container_client(container)
    container_client.upload_blob(
        blob_name,
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        overwrite=True,
        content_type="application/json",
    )