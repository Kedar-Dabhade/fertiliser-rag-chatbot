import os
from opensearchpy import OpenSearch
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables from .env if present
load_dotenv()

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "n864pyAkop5E5WB")
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "opensearch_fert")

# Since SSL is disabled for local dev
USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"

parsed = urlparse(OPENSEARCH_HOST)
host = parsed.hostname or "localhost"
port = parsed.port or 9200

client = OpenSearch(
    hosts=[{"host": host, "port": port}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=USE_SSL,
    verify_certs=VERIFY_CERTS
)

response = client.search(
    index=INDEX_NAME,
    body={"size": 10}
)

print(f"First 10 documents in index '{INDEX_NAME}':\n")
for i, hit in enumerate(response["hits"]["hits"], 1):
    print(f"--- Document {i} ---")
    print("ID:", hit["_id"])
    source = hit["_source"]
    for key, value in source.items():
        if key == "vector_field":
            print(f"{key}: [vector of length {len(value)}]")
        else:
            print(f"{key}: {value}")
    print() 