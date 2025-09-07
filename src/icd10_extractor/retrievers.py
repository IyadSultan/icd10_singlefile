"""
Retriever setup and management for FAISS and BM25.
"""

import os
import logging
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)


class RetrieverManager:
    """Manages FAISS and BM25 retrievers for ICD-10 data."""
    
    def __init__(self, icd10_csv_path: str, embeddings: OpenAIEmbeddings):
        """
        Initialize retriever manager.
        
        Args:
            icd10_csv_path: Path to the ICD-10 CSV file
            embeddings: OpenAI embeddings instance
        """
        self.icd10_csv_path = icd10_csv_path
        self.embeddings = embeddings
        self.vectorstore = None
        self.retriever = None
        self.bm25_retriever = None
        
    def setup_faiss_database(self, faiss_db_path: str = "data/icd10_faiss_db"):
        """Setup FAISS vector database for RAG."""
        if os.path.exists(faiss_db_path):
            logger.info(f"Found existing FAISS database at '{faiss_db_path}'")
            try:
                self.vectorstore = FAISS.load_local(
                    faiss_db_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                )
                logger.info("Successfully loaded existing FAISS database!")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing database: {e}")
        
        # Create new database
        logger.info("Creating new FAISS database...")
        try:
            icd10_df = pd.read_csv(self.icd10_csv_path)
            logger.info(f"Loaded {len(icd10_df)} ICD-10 codes from CSV")
            
            documents = []
            for index, row in icd10_df.iterrows():
                code = row.get('sub-code', '')
                definition = row.get('definition', '')
                
                page_content = f"ICD-10 Code: {code}\nDescription: {definition}"
                metadata = {
                    'code': code,
                    'definition': definition,
                    'row_index': index
                }
                
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
            
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Ensure data directory exists
            os.makedirs(os.path.dirname(faiss_db_path), exist_ok=True)
            self.vectorstore.save_local(faiss_db_path)
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            
            logger.info("FAISS vector database created and saved successfully!")
            
        except FileNotFoundError:
            logger.error(f"ICD-10 CSV file not found: {self.icd10_csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error creating FAISS database: {e}")
            raise
    
    def setup_bm25_retriever(self):
        """Setup BM25 retriever."""
        try:
            icd10_df = pd.read_csv(self.icd10_csv_path)
            texts = icd10_df.apply(
                lambda row: f"Code: {row.get('sub-code', '')}, Description: {row.get('definition', '')}", 
                axis=1
            ).tolist()
            
            self.bm25_retriever = BM25Retriever.from_texts(texts)
            logger.info("BM25 retriever created successfully")
            
        except Exception as e:
            logger.error(f"Error creating BM25 retriever: {e}")
            raise
