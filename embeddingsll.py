import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import pandas as pd

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
index_name = "all-products"  # or your actual index name

# Initialize Pinecone client and get/create index
pc = Pinecone(api_key=pinecone_api_key)
if not pc.list_indexes().names().__contains__(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )
index = pc.Index(index_name)

# Set up embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Read enriched data
df = pd.read_excel("products_enriched_final2.xlsx")

# Specify metadata columns
META_COLS = [
    "Name", "Price", "URL", "Nutrient_N", "Nutrient_P", "Nutrient_K", "Nutrient_S", "Nutrient_Ca", "Nutrient_Mg"
]

# Build documents
docs = []
for _, row in df.iterrows():
    # Build embedding text (include metadata fields in text for semantic search)
    embedding_text = f"""
Name: {row['Name']}
Price: {row['Price']}
Unit: {row.get('Unit', '')}
URL: {row['URL']}
Description: {row.get('Description', '')}
Benefits: {row.get('Benefits', '')}
Application & Advice: {row.get('Application & Advice', '')}
Storage: {row.get('Storage', '')}
Safety, Mixing and Compatibility: {row.get('Safety, Mixing and Compatibility', '')}
Nutrient_N: {row.get('Nutrient_N', '')}
Nutrient_P: {row.get('Nutrient_P', '')}
Nutrient_K: {row.get('Nutrient_K', '')}
Nutrient_S: {row.get('Nutrient_S', '')}
Nutrient_Ca: {row.get('Nutrient_Ca', '')}
Nutrient_Mg: {row.get('Nutrient_Mg', '')}
""".strip()
    # Build metadata dict
    metadata = {col: row.get(col, "") for col in META_COLS}
    docs.append(Document(page_content=embedding_text, metadata=metadata))

# Create the vector store and add documents
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
vectorstore.add_documents(docs)

print(f"✅ Successfully embedded and uploaded {len(docs)} product docs to Pinecone.") 