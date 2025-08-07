import fitz  # PyMuPDF
import httpx
import time
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List
from app.core.config import settings

class RetrievalService:
    """
    A service class responsible for document ingestion from a local file,
    processing, and retrieval using Pinecone as the vector database.
    """
    
    INDEX_NAME = "hackathon-rag-index"
    EMBEDDING_DIMENSION = 384 # Based on the 'all-MiniLM-L6-v2' model

    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', reranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the RetrievalService, loading ML models and the Pinecone client.
        """
        print("Initializing RetrievalService with Pinecone...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker_model = CrossEncoder(reranker_model_name)

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=512,
            chunk_overlap=76  # Overlap in tokens (15% of 512)
        )
        
        # Initialize Pinecone client
        self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = None
        self.text_chunks: List[str] = []

        # Attach to existing index if present to avoid reingestion
        try:
            existing_indexes = self.pinecone.list_indexes().names()
            if self.INDEX_NAME in existing_indexes:
                self.index = self.pinecone.Index(self.INDEX_NAME)
                print(f"Using existing Pinecone index: {self.INDEX_NAME}")
            else:
                print(f"Pinecone index '{self.INDEX_NAME}' not found. You may need to preload it.")
        except Exception as e:
            print(f"Warning: Could not inspect Pinecone indexes: {e}")

        print("RetrievalService initialized successfully.")

    def ingest_and_process_pdf(self, pdf_path: str):
        """
        Reads a PDF from a local file path only, extracts text, chunks it,
        and upserts the embeddings into a new Pinecone index.
        """
        print(f"Ingesting PDF from local path: {pdf_path}")
        
        # Only handle local file paths for now
        try:
            full_text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    full_text += page.get_text()
        except Exception as e:
            raise ValueError(f"Could not read or process PDF from path: {pdf_path}. Error: {e}")

        self.text_chunks = self.text_splitter.split_text(full_text)
        print(f"Split text into {len(self.text_chunks)} chunks.")

        # 2. Create Vector Embeddings
        print("Creating embeddings for text chunks...")
        chunk_embeddings = self.embedding_model.encode(
            self.text_chunks, 
            show_progress_bar=True
        )

        # 3. Setup Pinecone Index
        print("Setting up Pinecone index...")
        if self.INDEX_NAME in self.pinecone.list_indexes().names():
            print(f"Deleting existing index: {self.INDEX_NAME}")
            self.pinecone.delete_index(self.INDEX_NAME)
        
        print(f"Creating new index: {self.INDEX_NAME}")
        self.pinecone.create_index(
            name=self.INDEX_NAME,
            dimension=self.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
        while not self.pinecone.describe_index(self.INDEX_NAME).status['ready']:
            time.sleep(1)

        self.index = self.pinecone.Index(self.INDEX_NAME)
        print("Pinecone index is ready.")

        # 4. Upsert vectors into Pinecone
        print("Upserting vectors to Pinecone...")
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(self.text_chunks, chunk_embeddings)):
            vectors_to_upsert.append({
                "id": str(i),
                "values": embedding.tolist(),
                "metadata": {"text": chunk}
            })
        
        self.index.upsert(vectors=vectors_to_upsert, batch_size=100)
        print(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone.")


    def search_and_rerank(self, query: str, top_k_retrieval: int = 20, top_n_rerank: int = 5) -> List[str]:
        """
        Performs a two-stage search using Pinecone for retrieval and a CrossEncoder for reranking.
        """
        if not self.index:
            raise RuntimeError("Document has not been ingested. Call ingest_and_process_pdf() first or preload the index.")

        query_embedding = self.embedding_model.encode([query]).tolist()
        
        query_response = self.index.query(
            vector=query_embedding,
            top_k=top_k_retrieval,
            include_metadata=True
        )
        
        if not query_response['matches']:
            print("Warning: No relevant chunks found in Pinecone for the query.")
            return []

        retrieved_chunks = [match['metadata']['text'] for match in query_response['matches']]

        rerank_pairs = [[query, chunk] for chunk in retrieved_chunks]
        scores = self.reranker_model.predict(rerank_pairs)
        
        scored_chunks = list(zip(scores, retrieved_chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        reranked_chunks = [chunk for score, chunk in scored_chunks[:top_n_rerank]]
        
        return reranked_chunks
