"""Quick script to dump the full indexed content of the lunch menu PDFs."""
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()
endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
key = os.environ["AZURE_SEARCH_API_KEY"]
index = os.environ["AZURE_SEARCH_INDEX_NAME"]

client = SearchClient(endpoint, index, AzureKeyCredential(key))

# Try a filter-based search for source URLs containing menu PDFs
for file_num in ["979", "980", "981", "982", "977", "978"]:
    results = list(client.search(
        f"file={file_num}",
        top=5,
        select="content,source_url,page_title,last_modified",
    ))
    if results:
        print(f"\n=== file={file_num} ===")
        for r in results:
            print("URL:", r.get("source_url", ""))
            print("Last modified:", r.get("last_modified", ""))
            print("Content:\n", r.get("content", "")[:2000])
            print("-" * 60)

# Also search for catering page
print("\n=== Catering page search ===")
results = list(client.search("paella chorizo sausage hotdog roast", top=10, select="content,source_url,page_title"))
for r in results:
    print("URL:", r.get("source_url", ""))
    print("Content:", r.get("content", "")[:500])
    print("-" * 60)
