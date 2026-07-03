
import os

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import uuid
import numpy as np
import hashlib
from typing import List, Set, Dict, Tuple, Optional
from langchain_core.documents import Document
import logging
from difflib import SequenceMatcher

# Qdrant imports
from qdrant_client import QdrantClient, models

from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    SearchRequest,
    NamedVector,
    NamedSparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)
from transformers import AutoTokenizer
from pathlib import Path


import re
import pathlib
from pathlib import Path
import fasttext
import json


from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        TokenTextSplitter
    )
from transformers import AutoTokenizer

from langchain_core.documents import Document
    
    


# BGE-M3 for embeddings
from FlagEmbedding import BGEM3FlagModel

# LangChain Document
from langchain_core.documents import Document
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ Configuration ============
@dataclass
class QdrantRAGConfig:
    """Configuration for Qdrant-based RAG system"""
    
    # Qdrant settings
    QDRANT_PATH: str = "./qdrant_data" 
    COLLECTION_NAME: str = "machine_manuals"
    ServerMode: bool = False

    # BGE-M3 settings
    EMBED_MODEL_ID: str = "BAAI/bge-m3"
    USE_FP16: bool = True
    MAX_PASSAGE_LENGTH: int = 8192
    
    # Vector dimensions
    DENSE_VECTOR_SIZE: int = 1024
    
    
    # Hybrid search weights
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5
    

    
    # Batch settings
    BATCH_SIZE: int = 32
    
    # Index settings
    INDEX_DIR: str = "./qdrant_index"
    CACHE_DIR: str = "./qdrant_cache"
    
    # HNSW index parameters
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCT: int = 100
    
    # Payload index settings
    CREATE_PAYLOAD_INDEXES: bool = True

    # --- Dynamic retrieval sizing ---
    RETRIEVAL_RATIO: float = 0.25   # fraction of filtered points to prefetch
    RETRIEVAL_FLOOR_RATIO: float = 0.15           # min = 15% of filtered set
    RETRIEVAL_CEILING_RATIO: float = 0.80         # max = 80% of filtered set
    RETRIEVAL_ABS_MIN_K: int = 10                 # hard floor for tiny machines (seedcount=37)
    RETRIEVAL_ABS_MAX_K: int = 300                # hard ceiling to avoid runaway costs
    RRF_RATIO: float = 0.50
    FINAL_K_RATIO: float = 0.50    # prefetch_k * ratio = final rerank window
    MIN_FINAL_K: int = 10           # floor for final window
    # --------------------------------
    
    def __post_init__(self):
        Path(self.INDEX_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)

    def get_dynamic_retrieval_k(self, filtered_size: int) -> Tuple[int, int]:
        """
        Compute (prefetch_k, final_k) based on the number of points
        that actually match the routing filter — not the total collection size.

        prefetch_k  — candidates fed into RRF fusion per search arm
        final_k     — window returned after deduplication / reranking

        
        """
        prefetch_k = int(filtered_size * self.RETRIEVAL_RATIO)
        dynamic_floor   = max(self.RETRIEVAL_ABS_MIN_K,  int(filtered_size * self.RETRIEVAL_FLOOR_RATIO))
        dynamic_ceiling = min(self.RETRIEVAL_ABS_MAX_K,  int(filtered_size * self.RETRIEVAL_CEILING_RATIO))

        prefetch_k = max(dynamic_floor, min(prefetch_k, dynamic_ceiling))


        # Step 2: rrf_k from prefetch_k, capped below prefetch_k
        rrf_k = int(prefetch_k  * self.RRF_RATIO) # 50% . 0.25 dense + 0.25 sparse = 0.7 total fed into RRF
        rrf_k = min(rrf_k, prefetch_k - 1)      # Qdrant hard constraint


        final_k = int(rrf_k * self.FINAL_K_RATIO)
        final_k = max(self.MIN_FINAL_K, final_k)
        final_k = min(final_k, rrf_k - 1)           # ← hard cap: never >= rrf_k
        
        assert prefetch_k > rrf_k > final_k, (
        f"Ordering violated: prefetch_k={prefetch_k} rrf_k={rrf_k} final_k={final_k}"
            )

        
        return prefetch_k, rrf_k, final_k


# ============ BGE-M3 Embedding Manager ============
class BGEM3EmbeddingManager:
    """
    Manages BGE-M3 model for generating dense and sparse embeddings.
    
    BGE-M3 can generate three types of embeddings in a single pass:
    1. Dense vectors (1024 dimensions) - for semantic similarity
    2. Sparse vectors (lexical weights) - for keyword matching
    3. ColBERT vectors - for fine-grained matching (optional)
    """
    
    def __init__(self, config: QdrantRAGConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load BGE-M3 model"""
        logger.info(f"Loading BGE-M3 model: {self.config.EMBED_MODEL_ID}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        self.model = BGEM3FlagModel(
            self.config.EMBED_MODEL_ID,
            use_fp16=self.config.USE_FP16 and device == 'cuda',
            device=device
        )
        
        logger.info("BGE-M3 model loaded successfully")
    
    def encode(
        self,
        texts: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False,
        batch_size: int = 32,
        max_length: int = 8192
    ) -> Dict[str, Any]:
        """
        Encode texts to get dense and sparse embeddings.
        
        Args:
            texts: List of texts to encode
            return_dense: Whether to return dense vectors
            return_sparse: Whether to return sparse vectors
            return_colbert: Whether to return ColBERT vectors
            batch_size: Batch size for encoding
            max_length: Maximum token length
        
        Returns:
            Dictionary with:
            - 'dense_vecs': numpy array of shape (n, 1024) if return_dense
            - 'lexical_weights': list of dicts {token_id: weight} if return_sparse
            - 'colbert_vecs': list of numpy arrays if return_colbert
        """
        output = self.model.encode(
            texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
            batch_size=batch_size,
            max_length=max_length
        )
        
        return output
    
    def encode_single(
        self,
        text: str,
        return_dense: bool = True,
        return_sparse: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[int, float]]]:
        """
        Encode a single text and return dense and sparse vectors.
        
        Returns:
            Tuple of (dense_vector, sparse_dict)
            - dense_vector: numpy array of shape (1024,)
            - sparse_dict: {token_id: weight} for non-zero weights
        """
        output = self.encode(
            [text],
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert=False,
            batch_size=1
        )
        
        dense_vec = output['dense_vecs'][0] if return_dense else None
        sparse_dict = output['lexical_weights'][0] if return_sparse else None
        
        return dense_vec, sparse_dict
    
    def convert_sparse_to_qdrant(
        self,
        lexical_weights: Dict
    ) -> Tuple[List[int], List[float]]:
        """
        Convert BGE-M3 lexical weights to Qdrant sparse vector format.
        
        BGE-M3 returns: {token_id: weight, ...}
        Qdrant needs: SparseVector(indices=[...], values=[...])
        
        Args:
            lexical_weights: Dictionary of {token_id: weight}
        
        Returns:
            Tuple of (indices, values) for Qdrant SparseVector
        """
        indices = [int(token_id) for token_id in lexical_weights.keys()]
        values = [float(weight) for weight in lexical_weights.values()]
        
        # Sort by indices for consistency
        sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
        indices = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]
        
        return indices, values


# ============ Qdrant Store Manager ============
class QdrantStoreManager:
    """
    Manages Qdrant vector store with hybrid search capabilities.
    """
    
    def __init__(self, config: QdrantRAGConfig):
        self.config = config
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Qdrant"""
        if self.config.QDRANT_PATH:
            print(f"Connecting to local Qdrant at: {self.config.QDRANT_PATH}")
            logger.info(f"Connecting to local Qdrant at: {self.config.QDRANT_PATH}")
            self.client = QdrantClient(path=self.config.QDRANT_PATH)
        else:
            if self.config.ServerMode:
                qdrant_host = os.getenv("QDRANT_HOST", "localhost")
                qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
                logger.info(f"Connecting to Qdrant via REST at: "
                           f"{qdrant_host}:{qdrant_port}")
                self.client = QdrantClient(
                    host=qdrant_host,
                    port=qdrant_port
                )
        
        logger.info("Connected to Qdrant successfully")
    
    def collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.config.COLLECTION_NAME for c in collections)
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
            return False
    
    def create_collection(self, recreate: bool = False):
        """
        Create collection with dense and sparse vector support.
        
        Args:
            recreate: If True, delete existing collection and create new one
        """
        if recreate and self.collection_exists():
            logger.warning(f"Deleting existing collection: {self.config.COLLECTION_NAME}")
            self.client.delete_collection(self.config.COLLECTION_NAME)
        
        if self.collection_exists():
            logger.info(f"Collection {self.config.COLLECTION_NAME} already exists")
            return
        
        logger.info(f"Creating collection: {self.config.COLLECTION_NAME}")
        
        self.client.create_collection(
            collection_name=self.config.COLLECTION_NAME,
            
            vectors_config={
                "dense": VectorParams(
                    size=self.config.DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=self.config.HNSW_M,
                        ef_construct=self.config.HNSW_EF_CONSTRUCT
                    )
                )
            },
            
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
            
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
            )
        )
        
        if self.config.CREATE_PAYLOAD_INDEXES:
            self._create_payload_indexes()
        
        logger.info(f"Collection {self.config.COLLECTION_NAME} created successfully")
    
    def _create_payload_indexes(self):
        """Create payload indexes for efficient filtering"""
        
        self.client.create_payload_index(
            collection_name=self.config.COLLECTION_NAME,
            field_name="machine",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        self.client.create_payload_index(
            collection_name=self.config.COLLECTION_NAME,
            field_name="language",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            collection_name=self.config.COLLECTION_NAME,
            field_name="access_control",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            collection_name=self.config.COLLECTION_NAME,
            field_name="table_id",
            field_schema=models.PayloadSchemaType.KEYWORD,  # table_id is a string/keyword
        )
        
        logger.info("Payload indexes created")
    
    def upsert_points(
        self,
        points: List[PointStruct],
        batch_size: int = 100
    ):
        """
        Upsert points in batches.
        
        Args:
            points: List of PointStruct objects
            batch_size: Number of points per batch
        """
        total = len(points)
        
        for i in range(0, total, batch_size):
            batch = points[i:i + batch_size]
            
            self.client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=batch,
                wait=True
            )
            
            logger.info(f"Upserted batch {i // batch_size + 1}: "
                       f"{min(i + batch_size, total)}/{total} points")
    
    def get_collection_info(self) -> Optional[Dict]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.config.COLLECTION_NAME)
            logger.info(f"Collection info: {info}")
            return None
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None


# ============ Hybrid Retriever with Qdrant ============
class QdrantHybridRetriever:
    """
    Hybrid retriever combining dense and sparse search with RRF fusion.
    
    Uses Qdrant's prefetch + fusion API for efficient hybrid search:
    1. Dense search (semantic similarity)
    2. Sparse search (keyword matching)
    3. RRF fusion to combine results

    prefetch_k and final_k are computed dynamically from the number of
    points that survive the routing filter — not the total collection size.
    This gives tighter, more accurate candidate windows per machine/language.
    """
    
    def __init__(
        self,
        store_manager: QdrantStoreManager,
        embedding_manager: BGEM3EmbeddingManager,
        config: QdrantRAGConfig
        ):
        self.store = store_manager
        self.embeddings = embedding_manager
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = None,
        filter_machine: Optional[str] = None,
        filter_language: Optional[str] = None,
        filter_access_control: Optional[str] = None,
        use_hybrid: bool = True
        ) -> List[Document]:
        """
        Perform hybrid search combining dense and sparse retrieval.

        prefetch_k is derived from the number of points that match the
        routing filter (post-routing size), so it automatically adapts to
        how many chunks belong to the requested machine/language combination.

        Args:
            query: Search query
            top_k: Explicit override for final result count.
                   When None, final_k is derived dynamically.
            filter_machine: Optional machine name filter
            filter_language: Optional language filter
            filter_access_control: Optional access control filter
            use_hybrid: Whether to use hybrid search or dense only

        Returns:
            List of Document objects with content and metadata
        """
        # Build filter first so we can count matching points
        filter_conditions = self._build_filter(
            filter_machine, filter_language, filter_access_control
        )

        # Count points that survive routing — this is the key insight:
        # size AFTER filtering is what matters, not the full collection size.
        filtered_size = self._get_filtered_count(filter_conditions)
        prefetch_k, rrf_k, dynamic_final_k = self.config.get_dynamic_retrieval_k(filtered_size)
        final_k = dynamic_final_k

        logger.info(
            "search() — filtered_size=%d  prefetch_k=%d rrf_k =%d final_k=%d  query=%r",
            filtered_size, prefetch_k, rrf_k, final_k, query
        )

        # Encode query
        dense_vec, sparse_weights = self.embeddings.encode_single(
            query,
            return_dense=True,
            return_sparse=True
        )

        if use_hybrid:
            logger.info(f"using hybrid search")
            documents = self._hybrid_search(
                dense_vec=dense_vec,
                sparse_weights=sparse_weights,
                prefetch_limit=prefetch_k,
                rrf_k = rrf_k,
                filter_conditions=filter_conditions
            )
        else:
            logger.info(f"using dense search")
            documents = self._dense_search(dense_vec, prefetch_k, filter_conditions) # dense retrieveal using prefetch limit
            
        
        return documents, filtered_size, prefetch_k, rrf_k, final_k 

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def _get_filtered_count(self, filter_conditions: Optional[Filter]) -> int:
        """
        Return the number of points that match the routing filter.

        Uses Qdrant's count() with exact=False for an approximate answer
        in O(1) time — negligible overhead before the actual search.
        Falls back to collection count if the call fails.
        """
        try:
            result = self.store.client.count(
                collection_name=self.config.COLLECTION_NAME,
                count_filter=filter_conditions,
                exact=False   # approximate is fine for sizing
            )
            count = result.count
            logger.info("Filtered point count (approximate): %d", count)
            return count if count > 0 else 1   # avoid zero-division in ratio
        except Exception as e:
            logger.warning(
                "Could not get filtered count (%s) — using fallback collection count ", e
            )
            
            return self.store.client.get_collection(self.config.COLLECTION_NAME).points_count

    def _build_filter(
        self,
        machine: List[str],
        language: List[str],
        access_control: Optional[str]
        ) -> Optional[Filter]:

        must_conditions = []

        if machine:
            if isinstance(machine, str):
                machine = [machine]
            #machine.append('general') # to always add general section
            machine_conditions = [
                FieldCondition(
                    key="machine",
                    match=MatchValue(value = value.lower())
                )
                for value in machine
            ]
            must_conditions.append(Filter(should=machine_conditions))

        if language:
            if isinstance(language, str):
                language = [language]
            if len(language) == 1:
                must_conditions.append(
                    FieldCondition(
                        key="language",
                        match=MatchValue(value=language[0].lower())
                    )
                )
            else:
                language_conditions = [
                    FieldCondition(
                        key="language",
                        match=MatchValue(value=lang.lower())
                    )
                    for lang in language
                ]
                must_conditions.append(Filter(should=language_conditions))

        if access_control:
            must_conditions.append(
                FieldCondition(
                    key="access_control",
                    match=MatchValue(value=access_control.lower())
                )
            )

        if must_conditions:
            logger.info("Constructed filter conditions: %s", must_conditions)
            return Filter(must=must_conditions)

        return None

    def filter_has_matches(self, filter_conditions) -> bool:
        probe = self.store.client.scroll(
            collection_name=self.config.COLLECTION_NAME,
            filter=filter_conditions,
            limit=1,
            with_payload=False,
            with_vectors=False
        )
        return len(probe[0]) > 0

    # ------------------------------------------------------------------
    # Core search methods
    # ------------------------------------------------------------------

    def _hybrid_search(
        self,
        dense_vec: np.ndarray,
        sparse_weights: Dict,
        prefetch_limit: int,
        rrf_k : int,
        filter_conditions: Optional[Filter]
        ) -> List[Document]:
        """
        Hybrid search: dense + sparse prefetch → RRF fusion.

        prefetch_limit and final_top_k are computed by the caller from
        the post-routing filtered count, so they are always proportional
        to the actual candidate pool size.
        """
        sparse_indices, sparse_values = self.embeddings.convert_sparse_to_qdrant(
            sparse_weights
        )

        def _run_query(with_filter: Optional[Filter]):
            logger.info(f"rrf k {rrf_k}")
            return self.store.client.query_points(
                collection_name=self.config.COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=dense_vec.tolist(),
                        using="dense",
                        limit=prefetch_limit,
                        filter=with_filter
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        ),
                        using="sparse",
                        limit=prefetch_limit,
                        filter=with_filter
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit= rrf_k,       # fetch generously, then slice after dedup
                query_filter=with_filter,
                with_payload=True,
                with_vectors=False
            )

        results = _run_query(filter_conditions)

        if not results.points:
            logger.warning(
                "Filtered search returned 0 points — falling back to full collection"
            )
            results = _run_query(None)

        logger.info("Qdrant search returned %d raw points", len(results.points))

        # comvert, Add table links → deduplicate 
        documents = self._results_to_documents(results.points)
        logger.info(f'retriever chunk ids : len - {len(documents)}')
        for doc in documents:
            logger.info(doc.metadata['chunk_index'])
        return documents

    def _dense_search(
        self,
        dense_vec: np.ndarray,
        top_k: int,
        filter_conditions: Optional[Filter]
    ) -> List:
        """Perform dense-only search"""
        def _run_query(with_filter: Optional[Filter]):
            return self.store.client.query_points(
                        collection_name=self.config.COLLECTION_NAME,
                            query=dense_vec.tolist(),
                            using="dense",
                            limit=top_k,
                            query_filter=filter_conditions,
                            with_payload=True,
                            with_vectors=False
                        )

        results = _run_query(filter_conditions)

        if not results.points:
            logger.warning(
                "Filtered search returned 0 points — falling back to full collection"
            )
            results = _run_query(None)

        logger.info("Qdrant search returned %d raw points", len(results.points))
        documents = self._results_to_documents(results.points)

        return documents
    def _sparse_search(
        self,
        sparse_weights: Dict,
        top_k: int,
        filter_conditions: Optional[Filter]
    ) -> List:
        """Perform sparse-only search"""
        sparse_indices, sparse_values = self.embeddings.convert_sparse_to_qdrant(
            sparse_weights
        )
        
        results = self.store.client.query_points(
            collection_name=self.config.COLLECTION_NAME,
            query=SparseVector(
                indices=sparse_indices,
                values=sparse_values
            ),
            using="sparse",
            limit=top_k,
            query_filter=filter_conditions,
            with_payload=True,
            with_vectors=False
        )
        
        return results.points

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _deduplicate_by_chunkids(self, documents: List[Document]) -> List[Document]:
        """
        Remove duplicates based on chunk index.
        
        Useful when the same chunk appears multiple times due to
        different search strategies (dense vs sparse).
        """
        seen_sources: Dict[int, Document] = {}
        unique_docs = []
        logger.info(f"Number of docs before deduplication {len(documents)}")
        chunk_ids = []
        for doc in documents:
            chunk_idx = doc.metadata.get('chunk_index', -1)
            chunk_ids.append(chunk_idx)
            if chunk_idx not in seen_sources:
                seen_sources[chunk_idx] = doc
                unique_docs.append(doc)
            else:
                # Keep the chunk with the highest score
                existing_score = seen_sources[chunk_idx].metadata.get('qdrant_score', 0)
                new_score = doc.metadata.get('qdrant_score', 0)
                    
                if new_score > existing_score:
                    idx = unique_docs.index(seen_sources[chunk_idx])
                    unique_docs[idx] = doc
                    seen_sources[chunk_idx] = doc
        
        logger.info(f"chunk index: \n {chunk_ids}")
        logger.info(f"Number of docs after deduplication {len(unique_docs)}")
        return unique_docs
  


    def fetch_table_chunks(self, table_ids: list[str]) -> list[dict]:
        scroll_filter = Filter(
            must=[FieldCondition(key="table_id", match=MatchAny(any=table_ids))]
        )
        
        all_results = []
        offset = None                        # Qdrant uses point-ID cursor

        while True:
            batch, next_offset = self.store.client.scroll(
                collection_name=self.config.COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=100,                   # page size
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_results.extend(batch)

            if next_offset is None:          # no more pages
                break
            offset = next_offset

        logger.info(f"fetch_table_chunks: fetched {len(all_results)} chunks for {len(table_ids)} table_ids")
        return all_results

    def _results_to_documents(self, results: List) -> List[Document]:
        """Convert Qdrant results to LangChain Documents, then deduplicate"""
        documents = []
        
        for point in results:
            payload = point.payload
            
            doc = Document(
                page_content=payload.get("content", ""),
                metadata={
                    "source": payload.get("source", ""),
                    "section": payload.get("section", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "machine": payload.get("machine", ""),
                    'language': payload.get("language", ""),
                    'access_control': payload.get("access_control", ""),
                    "qdrant_score": point.score,
                    "qdrant_id": str(point.id),
                    
                    **payload.get("metadata", {})
                }
            )
            documents.append(doc)


        logger.info(f"Intial documents : {len(documents)}") 
       
        return self._deduplicate_by_chunkids(documents)
    
    def invoke(self, query: str) -> List[Document]:
        """LangChain-compatible interface"""
        return self.search(query)


# ============ Document Indexer ============
class QdrantDocumentIndexer:
    """
    Handles indexing documents into Qdrant with BGE-M3 embeddings.
    
    Process:
    1. Takes processed document chunks
    2. Generates dense and sparse embeddings using BGE-M3
    3. Creates Qdrant points with all data
    4. Uploads to Qdrant in batches
    """
    
    def __init__(
        self,
        store_manager: QdrantStoreManager,
        embedding_manager: BGEM3EmbeddingManager,
        config: QdrantRAGConfig
    ):
        self.store = store_manager
        self.embeddings = embedding_manager
        self.config = config
    
    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = None,
        show_progress: bool = True,
        force_reindex: bool = False
    ) -> int:
        """
        Index documents into Qdrant.
        
        Args:
            documents: List of Document objects to index
            batch_size: Number of documents to process per batch
            show_progress: Whether to show progress
            force_reindex: Force re-indexing even if documents exist
        
        Returns:
            Number of documents indexed
        """
        try:
            collection_info = self.store.client.get_collection(
                collection_name=self.config.COLLECTION_NAME
            )
            existing_count = collection_info.points_count
            
            if existing_count > 0 and not force_reindex:
                logger.info(f"✅ Collection already has {existing_count} documents. Skipping indexing.")
                logger.info("Use force_reindex=True to re-index anyway.")
                return existing_count
            elif existing_count > 0 and force_reindex:
                logger.warning(f"⚠️ Force re-indexing: Deleting {existing_count} existing documents...")
                self.store.client.delete_collection(collection_name=self.config.COLLECTION_NAME)
                self.store.create_collection(recreate=False)
                logger.info("Collection recreated, proceeding with indexing...")
                
        except Exception as e:
            logger.info(f"Collection doesn't exist or error checking: {e}. Proceeding with indexing...")
            self.store.create_collection(recreate=False)
        
        batch_size = batch_size or self.config.BATCH_SIZE
        total = len(documents)
        indexed = 0
        
        logger.info(f"🔄 Indexing {total} documents...")
        
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch_docs]
            
            embeddings = self.embeddings.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert=False,
                batch_size=batch_size
            )
            
            points = []
            for j, doc in enumerate(batch_docs):
                point_id = str(uuid.uuid4())
                dense_vec = embeddings['dense_vecs'][j]
                sparse_weights = embeddings['lexical_weights'][j]
                sparse_indices, sparse_values = self.embeddings.convert_sparse_to_qdrant(
                    sparse_weights
                )
                
                payload = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("file_name", ""),
                    "section": doc.metadata.get("section", ""),
                    "chunk_index": doc.metadata.get("chunk_id", i + j),
                    "machine": doc.metadata.get("machine_context", "").lower() ,
                    "language": doc.metadata.get("language", "").lower() ,
                    "access_control": doc.metadata.get("access_control", "").lower() ,
                    "table_id" : doc.metadata.get("table_id", ""),
                    "metadata": {
                        k: v for k, v in doc.metadata.items()
                        if k not in ["file_name", "section", "chunk_id", 
                                    "language", "access_control", "machine_context"]
                    }
                }
                
                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        )
                    },
                    payload=payload
                )
                points.append(point)
            
            self.store.client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=points,
                wait=True
            )
            
            indexed += len(points)
            
            if show_progress:
                logger.info(f"Indexed {indexed}/{total} documents")
        
        logger.info(f"✅ Indexing complete: {indexed} documents")
        return indexed


# ============ Integration with Your Existing RAG System ============
class QdrantRAGSystem:
    """
    Complete RAG system using Qdrant with BGE-M3 hybrid search.
    """
    
    def __init__(self, config: QdrantRAGConfig = None):
        self.config = config or QdrantRAGConfig()
        
        logger.info("Initializing Qdrant RAG System...")
        
        self.embeddings = BGEM3EmbeddingManager(self.config)
        self.store = QdrantStoreManager(self.config)
        self.retriever = QdrantHybridRetriever(
            self.store, self.embeddings, self.config
        )
        self.indexer = QdrantDocumentIndexer(
            self.store, self.embeddings, self.config
        )
        
        
        logger.info("Qdrant RAG System initialized")
    
    def initialize_collection(self, recreate: bool = False):
        """Create or recreate the collection"""
        self.store.create_collection(recreate=recreate)
    
    def index_documents(self, documents: List[Document], force_reindex) -> int:
        """Index documents into Qdrant"""
        if not self.store.collection_exists():
            self.initialize_collection()
        return self.indexer.index_documents(documents, force_reindex=force_reindex)
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return self.store.get_collection_info()


 