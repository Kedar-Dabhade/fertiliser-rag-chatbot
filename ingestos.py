import os
import pandas as pd
import numpy as np
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv
import re

# Load environment variables from .env if present
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FertilizerDataIngestion:
    def __init__(self):
        self.opensearch_url = os.getenv("OPENSEARCH_HOST", "https://localhost:9200")
        self.opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
        self.opensearch_password = os.getenv("OPENSEARCH_PASSWORD", "admin")
        self.opensearch_verify_certs = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
        self.index_name = os.getenv("OPENSEARCH_INDEX", "opensearch_fert")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

    def create_index_mapping(self):
        index_config = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "text": {"type": "text"},
                    "product_name": {"type": "keyword"},
                    "product_url": {"type": "keyword"},
                    "price": {"type": "float"},
                    "unit": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "nutrient_n": {"type": "float"},
                    "nutrient_p": {"type": "float"},
                    "nutrient_k": {"type": "float"},
                    "nutrient_s": {"type": "float"},
                    "nutrient_ca": {"type": "float"},
                    "nutrient_mg": {"type": "float"},
                    "created_at": {"type": "date"}
                }
            }
        }
        try:
            response = requests.get(
                f"{self.opensearch_url}/{self.index_name}",
                auth=(self.opensearch_user, self.opensearch_password),
                verify=self.opensearch_verify_certs
            )
            if response.status_code == 200:
                logger.info(f"Index '{self.index_name}' already exists")
                return True
            response = requests.put(
                f"{self.opensearch_url}/{self.index_name}",
                json=index_config,
                headers={"Content-Type": "application/json"},
                auth=(self.opensearch_user, self.opensearch_password),
                verify=self.opensearch_verify_certs
            )
            if response.status_code in [200, 201]:
                logger.info(f"Successfully created index '{self.index_name}'")
                return True
            else:
                logger.error(f"Failed to create index: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False

    def clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning and processing data...")
        df_clean = df.copy()
        df_clean = df_clean.fillna({
            'Name': 'Unknown Product',
            'Price': 0.0,
            'Unit': 'Unknown',
            'URL': '',
            'Description': 'No description available',
            'Benefits': 'Benefits not specified',
            'Application & Advice': 'Application advice not available',
            'Storage': 'Storage information not provided',
            'Safety, Mixing and Compatibility': 'Safety information not available',
            'Nutrient_N': 0.0,
            'Nutrient_P': 0.0,
            'Nutrient_K': 0.0,
            'Nutrient_S': 0.0,
            'Nutrient_Ca': 0.0,
            'Nutrient_Mg': 0.0
        })

        def clean_numeric(value):
            if pd.isna(value):
                return 0.0
            value = str(value)
            # Remove any non-numeric characters except dot and minus
            value = re.sub(r"[^0-9.\-]", "", value)
            try:
                return float(value)
            except ValueError:
                return 0.0

        nutrient_columns = ['Nutrient_N', 'Nutrient_P', 'Nutrient_K', 'Nutrient_S', 'Nutrient_Ca', 'Nutrient_Mg']
        for col in nutrient_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(clean_numeric)
        if 'Price' in df_clean.columns:
            df_clean['Price'] = df_clean['Price'].apply(clean_numeric)
        df_clean = df_clean.dropna(how='all')
        logger.info(f"Cleaned data: {len(df_clean)} products ready for ingestion")
        return df_clean

    def create_comprehensive_content(self, row: pd.Series) -> str:
        nutrients = []
        nutrient_mapping = {
            'Nutrient_N': 'Nitrogen (N)',
            'Nutrient_P': 'Phosphorus (P)',
            'Nutrient_K': 'Potassium (K)',
            'Nutrient_S': 'Sulfur (S)',
            'Nutrient_Ca': 'Calcium (Ca)',
            'Nutrient_Mg': 'Magnesium (Mg)'
        }
        for col, name in nutrient_mapping.items():
            if col in row and pd.notna(row[col]) and row[col] > 0:
                nutrients.append(f"{name}: {row[col]}%")
        nutrient_profile = ", ".join(nutrients) if nutrients else "Nutrient composition not specified"
        content = f"""Product Name: {row['Name']}
Price: ${row['Price']} per {row['Unit']}
Product Description: {row['Description']}
Key Benefits: {row['Benefits']}
Application and Usage Advice: {row['Application & Advice']}
Storage Instructions: {row['Storage']}
Safety and Compatibility Information: {row['Safety, Mixing and Compatibility']}
Nutrient Analysis: {nutrient_profile}
Product URL: {row['URL']}"""
        return content.strip()

    def create_metadata(self, row: pd.Series, index: int):
        return {
            'product_name': str(row['Name']),
            'product_url': str(row['URL']),
            'price': float(row['Price']),
            'unit': str(row['Unit']),
            'nutrient_n': float(row['Nutrient_N']),
            'nutrient_p': float(row['Nutrient_P']),
            'nutrient_k': float(row['Nutrient_K']),
            'nutrient_s': float(row['Nutrient_S']),
            'nutrient_ca': float(row['Nutrient_Ca']),
            'nutrient_mg': float(row['Nutrient_Mg']),
            'source': 'fertilizer_products.csv',
            'row_index': index,
            'created_at': datetime.now().isoformat()
        }

    def prepare_documents(self, df: pd.DataFrame):
        logger.info("Preparing documents for ingestion...")
        documents = []
        for index, row in df.iterrows():
            content = self.create_comprehensive_content(row)
            metadata = self.create_metadata(row, index)
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        logger.info(f"Prepared {len(documents)} documents")
        return documents

    def ingest_data(self, csv_file_path: str, batch_size: int = 10):
        try:
            logger.info("Step 1: Creating OpenSearch index...")
            if not self.create_index_mapping():
                return False
            logger.info("Step 2: Loading CSV data...")
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            df_clean = self.clean_and_process_data(df)
            documents = self.prepare_documents(df_clean)
            logger.info("Step 5: Creating vector store and ingesting data...")
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                if i == 0:
                    vector_store = OpenSearchVectorSearch.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        opensearch_url=self.opensearch_url,
                        index_name=self.index_name,
                        engine="faiss",
                        space_type="cosinesimil",
                        timeout=60,
                        http_auth=(self.opensearch_user, self.opensearch_password),
                        verify_certs=self.opensearch_verify_certs
                    )
                else:
                    vector_store.add_documents(batch)
                logger.info(f"Batch {batch_num} completed successfully")
            logger.info(f"✅ Successfully ingested {len(documents)} fertilizer products!")
            return True
        except Exception as e:
            logger.error(f"❌ Error during ingestion: {str(e)}")
            return False

    def verify_ingestion(self):
        try:
            response = requests.get(
                f"{self.opensearch_url}/{self.index_name}/_stats",
                auth=(self.opensearch_user, self.opensearch_password),
                verify=self.opensearch_verify_certs
            )
            if response.status_code == 200:
                stats = response.json()
                doc_count = stats['indices'][self.index_name]['total']['docs']['count']
                logger.info(f"Index contains {doc_count} documents")
                vector_store = OpenSearchVectorSearch(
                    index_name=self.index_name,
                    opensearch_url=self.opensearch_url,
                    embedding_function=self.embeddings,
                    engine="faiss",
                    space_type="cosinesimil",
                    http_auth=(self.opensearch_user, self.opensearch_password),
                    verify_certs=self.opensearch_verify_certs
                )
                test_results = vector_store.similarity_search("organic fertilizer", k=3)
                logger.info(f"Test search returned {len(test_results)} results")
                if test_results:
                    logger.info("Sample result:")
                    sample = test_results[0]
                    logger.info(f"  Product: {sample.metadata.get('product_name', 'Unknown')}")
                    logger.info(f"  Content preview: {sample.page_content[:100]}...")
                return True
            else:
                logger.error(f"Failed to get index stats: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error verifying ingestion: {str(e)}")
            return False

    def ingest_data_from_df(self, df: pd.DataFrame, batch_size: int = 10):
        """Ingest data from a DataFrame (used for Excel ingestion)."""
        try:
            logger.info("Step 1: Creating OpenSearch index...")
            if not self.create_index_mapping():
                return False
            logger.info("Step 2: Cleaning and processing DataFrame...")
            df_clean = self.clean_and_process_data(df)
            documents = self.prepare_documents(df_clean)
            logger.info("Step 3: Creating vector store and ingesting data...")
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                if i == 0:
                    vector_store = OpenSearchVectorSearch.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        opensearch_url=self.opensearch_url,
                        index_name=self.index_name,
                        engine="faiss",
                        space_type="cosinesimil",
                        timeout=60,
                        http_auth=(self.opensearch_user, self.opensearch_password),
                        verify_certs=self.opensearch_verify_certs
                    )
                else:
                    vector_store.add_documents(batch)
                logger.info(f"Batch {batch_num} completed successfully")
            logger.info(f"✅ Successfully ingested {len(documents)} fertilizer products!")
            return True
        except Exception as e:
            logger.error(f"❌ Error during ingestion: {str(e)}")
            return False

def main():
    """Run the ingestion process using environment variables for config. Now uses an Excel file by default."""
    # Update to use the provided Excel file as default
    excel_file_path = os.getenv("EXCEL_FILE", "products_enriched_final2.xlsx")
    ingestion = FertilizerDataIngestion()
    logger.info("🚀 Starting fertilizer data ingestion...")
    # Read Excel instead of CSV
    try:
        df = pd.read_excel(excel_file_path)
    except Exception as e:
        logger.error(f"❌ Failed to read Excel file: {str(e)}")
        return
    if ingestion.ingest_data_from_df(df, batch_size=10):
        logger.info("✅ Ingestion completed successfully!")
        if ingestion.verify_ingestion():
            logger.info("✅ Verification passed!")
        else:
            logger.warning("⚠️ Verification failed - check the logs")
    else:
        logger.error("❌ Ingestion failed - check the logs")

if __name__ == "__main__":
    main() 