import os
from opensearchpy import OpenSearch
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "n864pyAkop5E5WB")
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "opensearch_fert")

USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"

client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=USE_SSL,
    verify_certs=VERIFY_CERTS
)

if client.indices.exists(index=INDEX_NAME):
    client.indices.delete(index=INDEX_NAME)
    print(f"Deleted index '{INDEX_NAME}'")
else:
    print(f"Index '{INDEX_NAME}' does not exist") 