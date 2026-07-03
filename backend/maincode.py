
#importing python files for the RAG system
from text_processing import processDocument
from vectorstore import QdrantRAGConfig, QdrantRAGSystem

#importing other necessary python modules

import math
import torch.nn.functional as F
import json
import hashlib
from datetime import datetime
from logging import config
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import torch
from pathlib import Path
import logging
import re
import threading
import random
import time
import os
import numpy as np
import gc
import pickle
from collections import defaultdict
import fasttext
from transformers import LogitsProcessor, LogitsProcessorList
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from pydantic import Field
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat
from transformers import AutoTokenizer
lang_model = fasttext.load_model("lid.176.bin")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from typing import Set
from langchain_community.cache import InMemoryCache
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import langchain
from langdetect import detect
from transformers import BitsAndBytesConfig
from transformers import GenerationConfig
from datasets import Dataset
# Enable caching
langchain.llm_cache = InMemoryCache()
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ Enhanced Configuration ============
@dataclass
class AdvancedRAGConfig:
    """
    RAG system constants and settings are defined here.
    This includes model IDs, chunking parameters, retrieval settings, Qdrant connection info, LLM generation parameters, system settings, and feature flags.
    """


    EMBED_MODEL_ID: str = "BAAI/bge-m3"
    LLM_MODEL_ID:str = "Qwen/Qwen3-30B-A3B"
    RERANKER_MODEL_ID:str = "Qwen/Qwen3-Reranker-4B"


    # Chunking settings
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 80 #(15% of chunk size)

    # Retrieval settings
    TOP_K: int = 10
    RERANK_TOP_K: int = 10
    RERANK_MAXLENGTH: int = 1024

    # Qdrant settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_PATH: str = "./qdrant_data"  # For local storage
    COLLECTION_NAME: str = "machine_manuals"

    # LLM settings
    TEMP: float = 0.2
    TOP_P: float = 0.85
    TOP_K_SAMPLE : int = 20
    MAX_NEW_TOKENS: int = 1024
    USE_8BIT: bool = False
    USE_4BIT: bool = False
    REPEAT_PENALTY: float = 1.0

    # System settings
    USE_GPU: bool = True
    USE_QDRANT:bool = True
    BATCH_SIZE: int = 12
    MAX_WORKERS: int = 4
    CACHE_DIR: str = "./cache"
    LOG_DIR: str = "./logs"

    # Feature flags
    USE_HYBRID_SEARCH: bool = True
    USE_RERANKING: bool = True
    ENABLE_MONITORING: bool = True

    #ragas
    RUN_TYPE: str = ""

    def __post_init__(self):
        self.USE_GPU = torch.cuda.is_available()

# ============ Index Cache Manager ============
class IndexCacheManager:

    """
    Manages caching of vector store index to avoid reprocessing unchanged documents.
    store the number of files, their names, sizes, modification times, and hashes in a manifest file.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

        self.manifest_file = self.cache_dir / f"index_manifest.json"
        self.chunks_file = self.cache_dir / f"processed_chunks.json"

        # Ensure directories exist
    
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of a file for change detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""

    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata for change detection"""
        path = Path(file_path)
        return {
            "path": str(path.absolute()),
            "name": path.name,
            "size": path.stat().st_size,
            "modified_time": path.stat().st_mtime,
            "hash": self._compute_file_hash(file_path)
        }

    def _scan_documents_folder(self, folder_path: str, extensions: List[str] = None) -> Dict[str, Dict]:
        """Scan folder and get info for all documents"""
        if extensions is None:
            extensions = ['.pdf', '.md', '.csv', '.json']

        folder = Path(folder_path)
        files_info = {}

        if folder.is_file():
            # Single file provided
            files_info[str(folder)] = self._get_file_info(str(folder))
        else:
            # Directory provided
            for ext in extensions:
                for file_path in folder.glob(f"**/*{ext}"):
                    files_info[str(file_path)] = self._get_file_info(str(file_path))

        return files_info

    def _load_manifest(self) -> Optional[Dict]:
        """Load the cached manifest file"""
        if not self.manifest_file.exists():
            return None

        try:
            with open(self.manifest_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            return None

    def _save_manifest(self, files_info: Dict, config_hash: str):
        """Save manifest with current file info"""
        manifest = {
            "created_at": datetime.now().isoformat(),
            "config_hash": config_hash,
            "files": files_info,
            "file_count": len(files_info)
        }

        with open(self.manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved manifest with {len(files_info)} files")

    def _compute_config_hash(self, config) -> str:
        """Compute hash of relevant config settings that affect indexing"""
        config_str = f"{config.EMBED_MODEL_ID}_{config.CHUNK_SIZE_TOKENS}_{config.CHUNK_OVERLAP_TOKENS}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def save_processed_chunks(self, documents):
        """Save processed chunks to disk for hybrid search rebuilding"""
        try:
            serializable = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]

            with open(self.chunks_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")

    def load_processed_chunks(self) -> Optional[List[Document]]:
        """Load processed chunks from disk"""
        if not self.chunks_file.exists():
            return None

        try:
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                return [
                    Document(
                        page_content=item["page_content"],
                        metadata=item.get("metadata", {})
                    )
                    for item in data
                ]

        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return None

    def check_index_valid(self, documents_folder: str, config) -> Tuple[bool, str]:
        """
        Check if cached index is valid and up-to-date.

        Returns:
            Tuple of (is_valid, reason)
        """

        # Check if manifest exists
        manifest = self._load_manifest()
        if manifest is None:
            return False, "No manifest found"

        # Check config hash (embedding model, chunk settings changed?)
        current_config_hash = self._compute_config_hash(config)
        if manifest.get("config_hash") != current_config_hash:
            return False, "Configuration changed (embedding model or chunk settings)"

        # Scan current documents
        current_files = self._scan_documents_folder(documents_folder)
        cached_files = manifest.get("files", {})

        # Check for new files
        current_paths = set(current_files.keys())
        cached_paths = set(cached_files.keys())

        new_files = current_paths - cached_paths
        if new_files:
            return False, f"{len(new_files)} New files added: {[Path(f).name for f in list(new_files)[:3]]}"

        removed_files = cached_paths - current_paths
        if removed_files:
            return False, f" {len(removed_files)} Files removed: {[Path(f).name for f in list(removed_files)[:3]]}"

        # Check for modified files (by hash)
        for file_path, file_info in current_files.items():
            cached_info = cached_files.get(file_path, {})
            if file_info["hash"] != cached_info.get("hash"):
                return False, f"File modified: {file_info['name']}"

        # Check if chunks file exists (needed for hybrid search)
        if not self.chunks_file.exists():
            return False, "Processed chunks cache missing"

        return True, "Index is valid and up-to-date"

    def update_manifest(self, documents_folder: str, config):
        """Update manifest after rebuilding index"""
        files_info = self._scan_documents_folder(documents_folder)
        config_hash = self._compute_config_hash(config)
        self._save_manifest(files_info, config_hash)



class ChineseBanLogitsProcessor(LogitsProcessor):
    """
    Bans ALL Chinese character tokens from generation.

    Uses 3 detection methods to ensure complete coverage:
    1. Check decoded token output
    2. Check raw vocabulary string
    3. Check individual characters in vocab string
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._ban_mask = None

        # Build comprehensive Chinese character set for O(1) lookup
        self.chinese_chars = self._build_chinese_char_set()

        # Find all tokens to ban
        self.banned_token_ids = self._find_all_chinese_tokens()
        banned_ids = self.banned_token_ids  # whatever list you're building
        banned_decoded = [self.tokenizer.decode([tid]) for tid in banned_ids]

        # Check for accidental ASCII/JSON critical token banning
        critical = ['{', '}', '"', ':', 'Y', 'N', 'E', 'S', 'O', '\n', ' ']
        for c in critical:
            matches = []
            for tid, tok in zip(banned_ids, banned_decoded):
                # Strip SentencePiece space before checking
                tok_stripped = tok.replace('\u2581', '').lstrip(' ')
                if c in tok_stripped:
                    matches.append(tid)
            if matches:
                print(f"WARNING: '{c}' found in banned tokens: {matches[:5]}")

        logger.info(f"Banned tokens: {len(self.banned_token_ids)}")

        vocab_size = len(tokenizer)
        self._ban_mask = torch.zeros(vocab_size, dtype=torch.bool)
        valid_ids = [tid for tid in self.banned_token_ids if tid < vocab_size]
        if valid_ids:
           self._ban_mask[valid_ids] = True

    def _build_chinese_char_set(self) -> Set[str]:
        """
        Build a set of all Chinese characters for O(1) lookup.
        More reliable than regex for character-by-character checking.
        """
        chinese_chars = set()

        # CJK Unified Ideographs (the main block - most common)
        for cp in range(0x4E00, 0x9FFF + 1):
            chinese_chars.add(chr(cp))

        # CJK Extension A
        for cp in range(0x3400, 0x4DBF + 1):
            chinese_chars.add(chr(cp))

        # CJK Compatibility Ideographs
        for cp in range(0xF900, 0xFAFF + 1):
            chinese_chars.add(chr(cp))

        # CJK Extension B (less common but still used)
        for cp in range(0x20000, 0x2A6DF + 1):
            chinese_chars.add(chr(cp))

        # Additional ranges for completeness
        ranges = [
            (0x2A700, 0x2B73F),  # Extension C
            (0x2B740, 0x2B81F),  # Extension D
            (0x2B820, 0x2CEAF),  # Extension E
            (0x2CEB0, 0x2EBEF),  # Extension F
            (0x30000, 0x3134F),  # Extension G
            (0x2E80, 0x2EFF),    # CJK Radicals Supplement
            (0x2F00, 0x2FDF),    # Kangxi Radicals
            (0x31C0, 0x31EF),    # CJK Strokes
        ]

        for start, end in ranges:
            for cp in range(start, end + 1):
                try:
                    chinese_chars.add(chr(cp))
                except:
                    pass

        return chinese_chars

    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains any Chinese character."""
        if not text:
            return False
        return any(char in self.chinese_chars for char in text)

    def _find_all_chinese_tokens(self) -> Set[int]:
        banned = set()
        vocab = self.tokenizer.get_vocab()


        SENTENCEPIECE_SPACE = '\u2581'

        for token_str, token_id in vocab.items():
            should_ban = False

            # Method 1
            if self._contains_chinese(token_str):
                stripped = token_str.replace(SENTENCEPIECE_SPACE, '')
                if not any(0x20 <= ord(c) <= 0x7E for c in stripped):
                    should_ban = True

            # Method 2
            if not should_ban:
                try:
                    decoded = self.tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False
                    )
                    decoded_stripped = decoded.replace(SENTENCEPIECE_SPACE, '').lstrip(' ')
                    if any(0x20 <= ord(c) <= 0x7E for c in decoded_stripped):
                        continue
                    if self._contains_chinese(decoded_stripped):
                        should_ban = True
                except:
                    pass

            # Method 3
            if not should_ban:
                try:
                    converted = self.tokenizer.convert_tokens_to_string([token_str])
                    converted_stripped = converted.replace(SENTENCEPIECE_SPACE, '').lstrip(' ')
                    if any(0x20 <= ord(c) <= 0x7E for c in converted_stripped):
                        continue
                    if self._contains_chinese(converted_stripped):
                        should_ban = True
                except:
                    pass

            if should_ban:
                banned.add(token_id)
        return banned

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply the ban by setting Chinese token logits to -inf."""
        device = scores.device
        vocab_size = scores.shape[-1]

        if not hasattr(self, '_debug_logged'):
            logger.info(f"[DEBUG] ChineseBanLogitsProcessor called! Device: {device}, Scores shape: {scores.shape}")
            self._debug_logged = True

        # Cache it to avoid repeated transfers
        if not hasattr(self, '_cached_device') or self._cached_device != device:
            self._ban_mask_device = self._ban_mask.to(device)
            self._cached_device = device

        # Handle vocab size mismatch if it occurs
        if vocab_size != self._ban_mask_device.shape[0]:
            # Create correctly sized mask
            new_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            valid_ids = [tid for tid in self.banned_token_ids if tid < vocab_size]
            if valid_ids:
                new_mask[valid_ids] = True
            self._ban_mask_device = new_mask

        # Apply mask with proper broadcasting for batch dimension
        if scores.dim() == 2:
            # Shape: (batch_size, vocab_size)
            mask_expanded = self._ban_mask_device.unsqueeze(0).expand_as(scores)
            scores = scores.masked_fill(mask_expanded, float('-inf'))
        else:
            scores = scores.masked_fill(self._ban_mask_device, float('-inf'))

        return scores


# ============ Reranking System Qwen3 ============
class DocumentReranker:
    """Rerank retrieved documents using Qwen3-Reranker (LLM-based, yes/no logit scoring)"""

    def __init__(self, model_name, max_length):

        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.instruction ="""\
The Query is a technical question. The Document can be a chunk of text or a table markdown from a machine manual.

Judge relevance based on whether the Document contains information — directly or indirectly — that helps answer the Query.
Text chunks and table markdowns should be evaluated equally.

Scoring guidance:
- High relevance: Document directly addresses the Query or contains the specific procedure, value, or explanation needed.
- Medium relevance: Document is related and provides useful context, even if it doesn't fully answer the Query.
- Low relevance: Document covers the same machine or section but contains no information useful for answering the Query.

Example (high relevance): Query asks how to reset a machine → Document describes the reset procedure steps.
Example (low relevance): Query asks how to reset a machine → Document only lists machine dimensions and weight.

If only part of the Document is relevant, this should still increase the relevance score.
The Document and Query may be in different languages; judge based on content, not language match.
                        """



        self.PREFIX = """\
<|im_start|>system\n
Judge whether the Document meets the requirements based on the Query and the Instruct provided.
Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n
<|im_start|>user\n"""


        self.SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


        logger.info(f"Loading Qwen3-Reranker: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map= "auto",  # loads directly onto GPU,
            ).eval()

        # Get yes/no token IDs once at init

        self.token_true_ids = [self.tokenizer.convert_tokens_to_ids("Yes"),
           self.tokenizer.convert_tokens_to_ids("yes")]
        self.token_False_ids = [self.tokenizer.convert_tokens_to_ids("No"),
           self.tokenizer.convert_tokens_to_ids("no")]

        self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        self.instruction_tokens = self.tokenizer.encode(self.instruction, add_special_tokens=False)

    def _format_pair(self, query: str, doc: str) -> str:

        doc_budget = self.max_length -( len(self.prefix_tokens) + len(self.suffix_tokens) + len(self.instruction_tokens))

        # Truncate doc tokens manually
        doc_tokens = self.tokenizer.encode(doc, add_special_tokens=False)
        if len(doc_tokens) > doc_budget:
            logger.info(f"Document truncated for reranking: original tokens={len(doc_tokens)}, budget={doc_budget} for doc: {doc[:100]}...")
            doc_tokens = doc_tokens[:doc_budget]
            doc = self.tokenizer.decode(doc_tokens)

        output = f"{self.PREFIX}<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}{self.SUFFIX}"
        return output

    @torch.no_grad()
    def _compute_scores(self, query: str, texts: List[str], batch_size: int = 4) -> List[float]:
        all_scores = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            pairs = [self._format_pair(query, t) for t in batch_texts]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            pred_ids = outputs.logits.argmax(dim=-1)

            # Score = softmax over [yes, no] logits at last token position
            last_logits = outputs.logits[:, -1, :]
            yes_logits = last_logits[:, self.token_true_ids].max(dim=-1).values   # (4,)
            no_logits  = last_logits[:, self.token_False_ids].max(dim=-1).values  # (4,)

            probs = F.softmax(torch.stack([yes_logits, no_logits], dim=1), dim=-1)  # (4, 2)

            scores = probs[:, 0]  # (4,) # only taking yes probabilityy

            all_scores.extend(scores)
            torch.cuda.empty_cache()
            gc.collect()
        return all_scores

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """Rerank documents based on query relevance"""
        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} documents for query: '{query}'")

        texts = [doc.page_content for doc in documents]
        scores = self._compute_scores(query, texts)

        doc_scores = list(zip(documents, scores))

        for doc, score in sorted(doc_scores, key=lambda x: x[1], reverse=True):
            logger.info(f"Score: {score:.4f} for document: {doc.page_content[:100]}...")


        threshold = 0.5
        # Sort descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        for doc, score in doc_scores:
            doc.metadata["rerank_score"] = float(score)

        valids  = [d for d, score in doc_scores if score >= threshold]

        # fallback: if no docs passed threshold, take top 2 docs by score anyway
        if not valids:
            logger.warning(f"No documents passed the relevance threshold of {threshold}. Falling back to top 3 documents by score.")
            valids = [d for d,_ in doc_scores[:3]]
            return valids

        k = min(top_k, len(valids)) # edge case: if fewer than top_k docs are above threshold, just return those that are
        reranked_docs = valids[:k]
        logger.info(f"Documents after reranking: {len(reranked_docs)}")
        return reranked_docs

    def unload(self):
        """Delete cached pipeline and free GPU memory."""
        logger.info("Unloading reranker")

        if self.model is None:
            return

        try:
            # Remove accelerate hooks before deletion (critical for device_map="auto")
            if hasattr(self.model, 'hf_device_map'):
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(self.model, recurse=True)

            self.model.cpu()  # Move all weights to CPU first, releases GPU tensors
            del self.model
            del self.tokenizer

        except Exception as e:
            logger.warning(f"Error during unload cleanup: {e}")
        finally:
            self.model = None
            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info(f"Reranker GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"Reranker GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# ============ Monitoring and Evaluation ============
class RAGMonitor:
    """Monitor and evaluate RAG system performance"""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics = []
        self.report = {}

    def log_query(self, query: str, answer: str, retrieved_docs: List[Document],
                  latency: float, metadata: Dict = None):
        """Log query and response details"""

        contexts = [doc.page_content for doc in retrieved_docs] if retrieved_docs else []

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "contexts": contexts,  # Added for RAGAS
            "num_retrieved": len(retrieved_docs),
            "latency": latency,
            "metadata": metadata or {},
            "answer_correctness": "unrated",
        }

        self.metrics.append(log_entry)



    def update_last_entry_correctness(self, correctness: str) -> bool:
        """Update correctness of last entry"""
        if not self.metrics:
            logger.warning("No metrics to update")
            return False

        # Update in memory
        self.metrics[-1]['answer_correctness'] = correctness

        logger.info(f"Updated last entry with correctness: {correctness}")
        return True

    def generate_and_write_report(self) -> dict:
        """Generate performance report and also write the final metric list to a file"""
        if not self.metrics:
            return {
                "total_queries": 0,
                "rated_queries": 0,
                "avg_latency": 0,
                "median_latency": 0,
                "avg_docs_retrieved": 0,
                "answer_correctness_distribution": {}
            }

        rated_metrics = [m for m in self.metrics if m.get('answer_correctness') != 'unrated']
        df = pd.DataFrame(self.metrics)
        df_all = df.drop_duplicates(subset=['timestamp'], keep='last')

        self.report = {
            "total_queries": len(df_all),
            "rated_queries": len(rated_metrics),
            "avg_latency": float(df_all["latency"].mean()) if "latency" in df_all.columns else 0,
            "median_latency": float(df_all["latency"].median()) if "latency" in df_all.columns else 0,
            "avg_docs_retrieved": float(df_all["num_retrieved"].mean()) if "num_retrieved" in df_all.columns else 0,
        }
        torch.cuda.empty_cache()
        if rated_metrics:
            df_rated = pd.DataFrame(rated_metrics)
            self.report["answer_correctness_distribution"] = df_rated["answer_correctness"].value_counts(normalize=True).to_dict()
        else:
            self.report["answer_correctness_distribution"] = {}

        report_file = self.log_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(self.report, f, indent=2)



        # removing duplicates
        dedup = {}
        for record in self.metrics:
            dedup[record["timestamp"]] = record  # overwrite → keeps last

        result = list(dedup.values())
        log_file = self.log_dir / f"queries_{datetime.now().strftime('%Y%m%d')}.jsonl" # writing the queries at end
        with open(log_file, "w", encoding="utf-8") as f:
            for metric in result:
                f.write(json.dumps(metric, ensure_ascii=False) + "\n")


        return self.report


class SimpleChatHistory:
    """Chat history that stores Q&A pairs and tracks machine context."""

    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.pending_question: Optional[str] = None
        self.current_machine_context: Optional[str] = None
        self.original_machine_name: Optional[str] = None

    def add_exchange(self, question: str, answer: str, machine_context: Optional[str] = None):
        """Add a Q&A exchange to history and update machine context"""
        self.history.append({
            "question": question.strip(),
            "answer": answer.strip(),
            "machine": machine_context
        })


        if machine_context:
            self.current_machine_context = machine_context

    def set_pending_question(self, question: str):
        """Store a question that needs machine clarification"""
        self.pending_question = question.strip()

    def get_pending_question(self) -> Optional[str]:
        """Get and clear the pending question"""
        question = self.pending_question
        self.pending_question = None
        return question

    def has_pending_question(self) -> bool:
        """Check if there's a pending question"""
        return self.pending_question is not None


    def get_history_string(self) -> str:
        """Get formatted history string for prompt"""
        if not self.history:
            return "No previous conversation."

        formatted = []
        for msg in self.history[-5:]:
            
            machine_info = f" [about {self.get_original_machine_name}]" # if machine context is available, add it to the user message for better clarity in the prompt. also subscripting the first element
            formatted.append(f"User: {msg['question']}{machine_info}")
            formatted.append(f"Assistant: {msg['answer']}")

        return "\n".join(formatted)

    def clear(self):
        """Clear chat history and machine context"""
        self.history = []
        self.pending_question = None
        self.current_machine_context = None
        logger.info("Chat history and machine context cleared")



# ============ Main Advanced RAG System ============
class AdvancedLangChainRAG:
    """Advanced RAG system with intent routing, machine context memory, and index caching"""

    def __init__(self, config: AdvancedRAGConfig):
        
        self.config = config
        self.processor = None
        self.reranker = None
        self.logit_processor = None
        self.monitor = RAGMonitor(config.LOG_DIR) if config.ENABLE_MONITORING else None
        self.detected_language: str = "de"  # FIXED: Default language

        self.embedtokenizer = None

        # Index cache manager
        self.cache_manager = IndexCacheManager(config.CACHE_DIR)
        self.chat_history = SimpleChatHistory()

        # Main document vectorstore and retriever
        self.vectorstore = None
        self.retriever = None
        self.processed_docs = None

        #llm seetings
        self.llm_instance = None
        self._llm_lock = threading.Lock()
        self.tokenizer = None
        self.llm_device: Optional[str] = None

        self._get_reranker() # loading the reranker at init to avoid doing it during the first user query, which can be slow

        for dir_path in [config.CACHE_DIR, config.LOG_DIR]:
            Path(dir_path).mkdir(exist_ok=True, parents=True)

    def initialize_llm(self, model_id):
        """Lazily initialize and cache the HuggingFace pipeline and record device info."""

        if getattr(self, "llm_instance", None) is not None:
            logger.info(f"Using cached LLM instance (device={self.llm_device})")
            return self.llm_instance

        with self._llm_lock:
            if getattr(self, "llm_instance", None) is not None:
                logger.info(f"Using cached LLM instance (device={self.llm_device})")
                return self.llm_instance

            logger.info(f"Initializing {model_id}...")
            t0 = time.time()
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast= True,
                device_map= "auto"
            )
            t1 = time.time()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {}

            if self.config.USE_GPU:
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True 
            else:
                model_kwargs["torch_dtype"] = torch.float32
                model_kwargs["device_map"] = {"": "cpu"}
           

            # quantization flags if requested
            quant_config = None

            if getattr(self.config, "USE_8BIT", False):
                logger.info('Model quantised in 8 bits')
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )

            elif getattr(self.config, "USE_4BIT", False):
                
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                logger.info('Model quantised in 4 bits')
            
            if quant_config:
                model_kwargs["quantization_config"] = quant_config

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            t4 = time.time()
            self.model.generation_config = GenerationConfig(
                        do_sample=True,
                        temperature=self.config.TEMP,
                        top_p=self.config.TOP_P,
                        top_k=self.config.TOP_K_SAMPLE,
                        max_new_tokens=self.config.MAX_NEW_TOKENS,
                        repetition_penalty=self.config.REPEAT_PENALTY,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
           

            self.chinese_logit_processor = ChineseBanLogitsProcessor(self.tokenizer)

            # Create a LogitsProcessorList
            self.logit_processor = LogitsProcessorList([self.chinese_logit_processor])

            # Build pipeline kwargs
            pipeline_device_arg = None
            if getattr(self.model, "hf_device_map", None) is None:
                if self.config.USE_GPU:
                    pipeline_device_arg = 0
                else:
                    pipeline_device_arg = -1

            pipeline_kwargs = dict(
                model=self.model,
                tokenizer=self.tokenizer,
                # Length
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                # Sampling
                do_sample = True,
                temperature=self.config.TEMP,
                top_p=self.config.TOP_P,
                top_k=self.config.TOP_K_SAMPLE,
                # Quality control
                repetition_penalty=self.config.REPEAT_PENALTY,
                # Tokens
                return_full_text=False,
                pad_token_id= self.tokenizer.pad_token_id,
                eos_token_id= self.tokenizer.eos_token_id,

            )


            if pipeline_device_arg is not None:
                pipeline_kwargs["device"] = pipeline_device_arg

            pipe = pipeline("text-generation", **pipeline_kwargs)
            logger.info("pipeline generation config:")
            for k, v in pipe.model.generation_config.to_dict().items():
                logger.info(f"{k}: {v}")

            try:
                pipe_device = getattr(pipe, "device", None)
                if pipe_device is not None:
                    if isinstance(pipe_device, torch.device):
                        self.llm_device = str(pipe_device)
                        self.llm_using_gpu = "cuda" in str(pipe_device).lower()
                    elif isinstance(pipe_device, int):
                        self.llm_device = f"cuda:{pipe_device}" if torch.cuda.is_available() else f"cpu:{pipe_device}"
                        self.llm_using_gpu = torch.cuda.is_available() and pipe_device >= 0
            except Exception:
                pass

            self.llm_instance = HuggingFacePipeline(
                                  pipeline=pipe )

            t5 = time.time()
            logger.info(f"LLM initialization times: Tokenizer={t1-t0:.2f}s, Model={t4-t1:.2f}s, Pipeline setup={t5-t4:.2f}s")

            logger.info(f"LLM initialized. Device: {self.llm_device} | Using GPU: {self.llm_using_gpu}")

    def _get_reranker(self):
        """Lazy load reranker only when needed"""
        if self.reranker is None:
            torch.cuda.empty_cache()  # clear cache first
            self.reranker = DocumentReranker(self.config.RERANKER_MODEL_ID, self.config.RERANK_MAXLENGTH)
            logger.info("Reranker loaded")

    def load_embedtokenizer(self):
        try:
            self.embedtokenizer = AutoTokenizer.from_pretrained(self.config.EMBED_MODEL_ID)
            logger.info(f"Loaded tokenizer for {self.config.EMBED_MODEL_ID}")
        except Exception as e:
            logger.info(f"Warning: Could not load tokenizer, falling back to word splitting: {e}")
            self.embedtokenizer = None

    def unload_embedtokenizer(self):
        """
        Unload the embedding tokenizer to free memory.
        """
        if getattr(self, "embedtokenizer", None) is not None:
            try:
                del self.embedtokenizer
                self.embedtokenizer = None
                gc.collect()
                logger.info("Embed tokenizer unloaded successfully.")
            except Exception as e:
                logger.info(f"Warning: Failed to unload tokenizer: {e}")
        else:
            logger.info("Embed tokenizer is already unloaded or was never initialized.")

    def initialize_embeddings(self, use_cache: bool = True):
        """Initialize embeddings with optional caching"""
        base_embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBED_MODEL_ID,
            model_kwargs={'device': 'cuda' if self.config.USE_GPU else 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 8 }
        )
        return base_embeddings

    def get_llm_device(self) -> str:
        """Return a human-readable string describing the device(s) used by the LLM."""
        return self.llm_device or ("cuda" if torch.cuda.is_available() else "cpu")

    def is_llm_using_gpu(self) -> bool:
        """Return True if the LLM is using a CUDA/GPU device for generation."""
        return getattr(self, "llm_using_gpu", torch.cuda.is_available())

    def unload_llm(self):
        """Delete cached pipeline and free GPU memory."""
        logger.info("unloading")
        with self._llm_lock:
            if self.llm_instance is None:
                return

            try:
                pip = getattr(self.llm_instance, "pipeline", self.llm_instance)

                # Move model to CPU first (forces CUDA tensors to release)
                if hasattr(pip, "model"):
                    pip.model.cpu()
                    del pip.model

                if hasattr(pip, "tokenizer"):
                    del pip.tokenizer

                # Delete the pipeline itself
                del pip

            except Exception as e:
                logger.warning(f"Error during unload cleanup: {e}")

            # Clear all references
            del self.llm_instance
            self.llm_instance = None
            self.tokenizer = None

            # Force Python garbage collection BEFORE clearing CUDA cache
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Optional: also release reserved memory back to OS
                torch.cuda.reset_peak_memory_stats()

            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logger.info(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    def setup_retriever(self, documents: List[Document], force_rebuild :bool):
        if self.config.USE_QDRANT:
            qdrant_config = QdrantRAGConfig(
                COLLECTION_NAME=self.config.COLLECTION_NAME
            )
            self.qdrant_system = QdrantRAGSystem(qdrant_config)
            self.qdrant_system.initialize_collection(recreate = force_rebuild)
            _ = self.qdrant_system.index_documents(documents, force_reindex=False)
            self.qdrant_system.get_stats()
            self.retriever = self.qdrant_system.retriever

  
    def build_prompt(self, question: str, context: str, history: str, machine_context: str = " ") -> str:
        """Build the prompt for the LLM"""

        logger.info(f'Detected language: {self.detected_language}')

        LANG_MAP = {
            # Major European languages
            "en": "English",
            "de": "German (Deutsch)",
            "fr": "French (Français)",
            "es": "Spanish (Español)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "nl": "Dutch (Nederlands)",
            "pl": "Polish (Polski)",
            "sv": "Swedish (Svenska)",
            "da": "Danish (Dansk)",
            "no": "Norwegian (Norsk)",
            "fi": "Finnish (Suomi)",
            "el": "Greek (Ελληνικά)",
            "cs": "Czech (Čeština)",
            "hu": "Hungarian (Magyar)",
            "ro": "Romanian (Română)",
            "bg": "Bulgarian (Български)",
            "sk": "Slovak (Slovenčina)",
            "sl": "Slovenian (Slovenščina)",
            "hr": "Croatian (Hrvatski)",
            "sr": "Serbian (Српски)",
            "bs": "Bosnian (Bosanski)",
            "mk": "Macedonian (Македонски)",

        }


        lang_name = LANG_MAP.get(self.detected_language, self.detected_language)

        system_msg = f"""\
You are a helpful technical assistant. The query is a technical question regarding a {machine_context} machine. The provided context are blocks of text from the {machine_context} instruction manuals.
You MUST answer in {lang_name} only, regardless of the language of the provided context.
You MUST follow the CRITICAL RULES, CITATION RULES, and LANGUAGE RULES defined below when answering the question. All rule groups are mandatory and must be applied together.

CRITICAL RULES:

1. The provided context is the sole source of factual information.
Do not use information from prior knowledge, assumptions, chat history, or external sources unless it also appears in the provided context.
Chat history may be used only to resolve references, pronouns, or omitted details in the current question.
2. Understand what the question is asking for conceptually, then check if the provided context covers that concept — the exact words in either the question or the provided context don't matter. 
If the user asks about 'tempering' and the provided context describes 'acclimating to room temperature,' these describe the same process. 
Match concepts rather than exact wording. Synonyms, paraphrases, alternative terminology, and process descriptions may be considered equivalent when they describe the same operation or concept. Note any terminology gap briefly if useful with source header information.
4. If information originates from a source section that has a chapter/section number, preserve that chapter/section number in the answer whenever reasonably possible.
5. Choose ONE of the following — never both:
   a) If the provided context contains no information that addresses the question → respond in {lang_name} with:
    "I am unable to assist you in this situation. Please get in touch with Company" This is the fallback message. Do NOT include any answer before or after this message.
   b) If the provided context contains enough information to fully or partially answer the question → answer it. Do NOT append the fallback message.
   When answering partially, answer only the portions supported by the provided context.Do not speculate or complete missing information using external knowledge.

CITATION RULES:

If information from the provided context is used, apply both of the following citation mechanisms:
a) Inline Citations (when required by the Inline Citation rules below).
b) Final Sources Section (when required by the Final Sources Section rules below).

# Inline Citations:

1. Inline citations are required only when the answer explicitly references a document structure element such as Chapter, Section, Subsection, Appendix, Table, Figure, or a similar numbered document reference.
2. Append the inline citation immediately after the referenced document structure element.
Inline citation format:
```
[Source: Source N]
```
Example:
```
The Auto Sample Cleaner is intended for designated use as described in Section 1.1 [Source: Source 3].
```
3. Do not add inline citations to factual statements, descriptions, procedures, warnings, summaries, or other content solely because the information originated from a source containing numbered document references.
Example:
```
The Auto Sample Cleaner is a stationary device used for determining total dockage, sorting, and automatically weighing grain crops.
```
4. If a sentence contains an explicit document structure reference (for example, a Chapter, Section, Subsection, Appendix, Table, or Figure reference), add an inline citation even if the remainder of the sentence contains factual information.
5. Inline citations are used exclusively for explicit document structure references. No other content should receive inline citations.

# Final Sources Section:

6. If one or more sources from the provided context were used to answer the question, include a complete list of all used sources at the end of the answer under a separate "Sources" section.
Format:
```
Sources:
Source 3 [17409081_BA_Auto Sample Cleaner_R6.2_12.02.2025_en, Page 5 - 7]
```
7. Include every source from which information was actually used in the generated answer.
8. Do not include retrieved sources whose content was not used.
9. Include each source only once in the final Sources section, even if it is referenced multiple times in the answer.
10. List sources in ascending numerical order of their source identifiers (Source 1, Source 2, Source 3, etc.).
11. Copy each source header exactly as it appears in the provided context. Do not translate, reformat, abbreviate, correct, normalize, or otherwise modify the source header.
12. The absence of inline citations does not remove the requirement to include a final Sources section if source content was used.
13. If the fallback message is used, do not include a Sources section.
14. The examples above use placeholder source headers for illustration only. Never reproduce example source headers unless they appear in the provided context. Always use the exact source identifier and source header from the provided context.


LANGUAGE RULES:

- Respond ONLY in {lang_name}, even if the provided context contains other languages.
-Source citations may contain original filenames and page references.
Everything else must be in {lang_name}.
- Never mix languages in your response.

REMEMBER: Your entire response MUST be in {lang_name} only.
"""


        user_content = f"""\
Context:
{context}

Question: {question}

Chat History:
{history}

Provide a clear and concise answer based on the provided context for the question.
REMEMBER: Your entire response MUST be in {lang_name} only.
"""

        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False, # we tokenize it later
                    add_generation_prompt=True,
                    enable_thinking= False
                )

                token_length = len(self.tokenizer(prompt)['input_ids'])
                llm_context_length = getattr(self.tokenizer, 'model_max_length', 'unknown')
                logger.info(f"[INFO] LLM context length: {llm_context_length} tokens")
                logger.info(f"[INFO] Built chat template prompt with {token_length} tokens")

                return prompt
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback format")

                prompt = f"""<|im_start|>system
                        /no_think
                        {system_msg}<|im_end|>
                        <|im_start|>user
                        {user_content}<|im_end|>
                        <|im_start|>assistant
                        """


        return prompt

    def detect_language_fasttext(self, text: str, threshold: float = 0.6) -> dict:
        labels, probs = lang_model.predict(text)
        lang = labels[0].replace("__label__", "")
        conf = probs[0]

        return {
                    "language": lang if conf >= threshold else "en",
                    "confidence": conf
                }


    def build_context(self, rerank_results, max_tokens: int = 12000) -> str:
        """
        Build context string from reranked documents with:
        - Token/char limit guard
        - Source metadata
        - Empty content filtering
        """
        context_parts = []
        total_tokens = 0

        for idx, doc in enumerate(rerank_results):
            content = doc.page_content.strip()

            # Skip empty chunks
            if not content:
                continue

            # Build source metadata  if available
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            source_label = metadata.get("source", f"Document {idx + 1}")
            page_start = metadata.get("page_start", None)
            page_end = metadata.get("page_end", None)

            header = f"Source {idx + 1} [{source_label}"
            header += f", Page {page_start} - {page_end}]:" if page_start else "]:"

            part = f"{header}\n{content}"

            # Guard against overflow
            if total_tokens + len(self.tokenizer(part)['input_ids']) > max_tokens:
                logger.warning(f"[Context] Truncating at source {idx + 1} — token limit reached")
                break

            context_parts.append(part)
            total_tokens += len(self.tokenizer(part)['input_ids'])

        return "\n\n".join(context_parts)

    def _rag_query(self, question, return_sources, start_time,
                   machine_context) :
        """Execute RAG pipeline for document-based queries"""

        if self.retriever is None:
            answer = "The document retrieval system is not initialized. Please load documents first."
            logger.warning(answer)
            return self._format_response(question, answer, [], time.time() - start_time,
                                        intent="document_query", used_rag=False)

        access_control = 'public'
        self.detected_language = self.detect_language_fasttext(question)["language"]

        if len(machine_context) == 1:
            logger.info(f'Machine context is one')
            raw_question = question.replace('?',' ').strip() # do a question preprocssing. remove the question mark.
            question = self.rephrase_query(raw_question, self.chat_history.original_machine_name) # if machine context is available, rephrase the query to be more specific to the machine, which can help retrieval. if original machine name is not available, use the raw question without rephrasing.
            if question is None or question.strip() == "":
                question = raw_question
            logger.info(f'question {question} machine context {machine_context} language {self.detected_language}')
            if machine_context[0] is None: # Fallback: if machine context is none, do retriever on entire dps
                docs, filtered_size, prefetch_k, rrf_k, final_k = self.retriever.search(
                                    query=question,
                                    top_k=None,             # None → fully dynamic based on filtered count
                                    filter_machine = None,
                                    filter_language= None,
                                    filter_access_control= access_control,
                                    use_hybrid=True
                                )

            else:
                docs, filtered_size, prefetch_k, rrf_k, final_k  = self.retriever.search(
                                    query= question,
                                    top_k= None,             # None → fully dynamic based on filtered count
                                    filter_machine = machine_context,
                                    filter_language= None,
                                    filter_access_control= access_control,
                                    use_hybrid=True
                                )

            if self.config.USE_RERANKING:
                if self.reranker is None:
                   self._get_reranker()
                rerank_results = self.reranker.rerank(question, docs, final_k)
                logger.info(f'docs reranked to top {len(rerank_results)} docs.')

            else:
                logger.info(f'Not using reranking.')
                rerank_results = docs
        else:
            logger.warning('Machine context is more than 1')
            rerank_results = []
            filtered_size, prefetch_k, rrf_k, final_k  = 0,0,0,0
        

        #-----now get the linked tables in the reranked docs if exist
        cachefilepath = self.cache_manager.chunks_file
        logger.info(f"cachefilepath: {cachefilepath}")
        with open(cachefilepath,"r") as f:
            cachedata = json.load(f)

        rerankedwithdocs = []
        for doc in rerank_results:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            table_links = metadata.get("table_links","")
            rerankedwithdocs.append(doc) # to have the same order
            if table_links:
                # fetech corresponding chunks from the cache file
                logger.info(f"table links found {table_links}")
                for tablelink in table_links:
                    for chunk in cachedata:
                        if chunk.get("metadata", {}).get("table_id","") == tablelink:
                            #convert tabledocs into doc object ad append it to
                            rerankedwithdocs.append(Document(page_content=chunk.get('page_content'), metadata= chunk.get("metadata", {})))
                            break
            else:
                logger.info(f"No table links found ")




        logger.info(f"len of reranked : {len(rerankedwithdocs)}")

        context = self.build_context(rerankedwithdocs)

        history_str = self.chat_history.get_history_string()
        
        prompt = self.build_prompt(question, context, history_str, machine_context = machine_context[0] if machine_context else " ")

        if prompt:
            logger.info(f'prompt is \n {prompt}')
            promptlength = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            logger.info(f'Prompt Length: {promptlength}')
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")

                device = next(self.llm_instance.pipeline.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self.llm_instance.pipeline.model.generate(
                            **inputs,
                            max_new_tokens=self.config.MAX_NEW_TOKENS,
                            do_sample= True,
                            temperature=self.config.TEMP,
                            top_p=self.config.TOP_P,
                            top_k=self.config.TOP_K_SAMPLE,
                            repetition_penalty=self.config.REPEAT_PENALTY,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            logits_processor=self.logit_processor,
                            )

                input_length = inputs['input_ids'].shape[1]
                new_tokens = outputs[0][input_length:]
                raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                logger.info(f"Raw LLM output type: {raw_response}")
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                raw_response = f"Error generating response: {str(e)}"
            if "<think>" in raw_response and not "</think>" in raw_response:
                logger.info(f"!!!!!!!Unclosed <tool_call> tag found in response .")

                raw_response = " "   # thinking tag is not closed, discard response to avoid parsing issues
            if "<think>" in raw_response and "</think>" in raw_response:
                logger.info(f"closed <tool_call> tag found in response")
                raw_response = raw_response.split("</think>", 1)[1].strip()

            clean_answer = raw_response.strip()
            if self.chinese_logit_processor._contains_chinese(clean_answer):
                logger.info(" Chinese ban not working!")
            else:
                logger.info("chinese ban working correctly.")
        else:
            logger.warning('prompt Constructor error')

        self.chat_history.add_exchange(question, clean_answer, machine_context)

        latency = time.time() - start_time

        if self.monitor:
            self.monitor.log_query(
                query=question,
                answer=clean_answer,
                retrieved_docs=rerank_results,
                latency=latency,
                metadata={
                    "machine_context": machine_context,
                    "detected_language": self.detected_language,
                    "access_control": access_control

                }
            )

        return self._format_response(
            question = question, answer = clean_answer,docs = rerank_results, latency = latency,
            intent="document_query", used_rag=True, return_sources=return_sources,
            machine_context=machine_context ,
            FS= filtered_size, PK = prefetch_k, RK = rrf_k, FK = final_k,
            prompt_length = promptlength,
            reranked_docs_len = (len(rerankedwithdocs))

        )

    def _format_response(self, question: str, answer: str,
                        latency: float, docs: List[Document],
                        FS:int = None, PK:int = None, RK:int = None,
                        FK:int = None,
                         prompt_length :int = None, reranked_docs_len:int = None,
                        intent: str = None, used_rag: bool = True,
                        return_sources: bool = True, awaiting_clarification: bool = False,
                        machine_context: Optional[str] = None,
                        ) -> Dict[str, Any]:
        """Format the response consistently"""
        response = {
            "question": question,
            "answer": answer,
            "latency": latency,
            "intent": intent,
            "used_rag": used_rag,
            "awaiting_clarification": awaiting_clarification,
            "machine_context": machine_context or self.chat_history.current_machine_context,
            'query_language': self.detected_language,
            'FS' : FS,
            'PK': PK,
            'RK' : RK,
            'FK' : FK,
            'prompt_length' : prompt_length ,
            'reranked_docslen' : reranked_docs_len
        }

        if return_sources and docs:
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]

        return response

    def clear_history(self):
        """Clear chat history and machine context"""
        self.chat_history.clear()
        self.machine_clarified = False # reset clarification flag

    def get_current_machine(self) -> Optional[str]:
        """Get the currently active machine context"""
        return self.chat_history.current_machine_context

    def set_current_machine(self, machine_name: str):
        """Manually set the current machine context"""
        self.chat_history.current_machine_context = machine_name
        logger.info(f"Current machine context set to: {machine_name}")
    
    def set_original_machine_name(self, original_machine_name: str):
        """Set the original machine name for query rephrasing fallback"""
        self.chat_history.original_machine_name = original_machine_name
        logger.info(f"Original machine name set to: {original_machine_name}")

    def rephrase_query(self, query, machine_name) :

        system_msg = f"""\
You are a query rewriting assistant for a technical documentation chatbot.

Your task is to rewrite the user's query into a standalone retrieval query for hybrid search.

Rules:
- If the query contains only generic references such as "it", "this", "that", "the machine", or "the device", and no prior context exists, rewrite the query as a request for information about the provided machine name itself.
- Preserve all technical terminology, model numbers, alarm codes, parameter names, and part numbers exactly.
- Use information from chat history only when it is necessary to resolve the user's question.
- Prioritize nouns and technical entities over conversational wording.
- Do NOT introduce information that is not supported by the query or chat history.
- If the query is already clear and standalone, you can keep it mostly unchanged.
- Return only the rewritten query in string format.""" 

        user_content = f"""\
Machine Name:
{machine_name}

Current Query:
{query}

Chat History:
{self.chat_history.get_history_string()}
"""
     

        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize= False, #we tokenize it later
                    add_generation_prompt= True,
                    enable_thinking= False
                )
    
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback format")

                prompt = f"""<|im_start|>system
                        /no_think
                        {system_msg}<|im_end|>
                        <|im_start|>user
                        {user_content}<|im_end|>
                
                        <|im_start|>assistant
                        """

        if prompt:
            logger.info(f'rephrased prompt is \n {prompt}')
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")

                device = next(self.llm_instance.pipeline.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self.llm_instance.pipeline.model.generate(
                            **inputs,
                            max_new_tokens=120,
                            do_sample=False,          # deterministic — important for rephrasing
                            temperature=None,
                            top_p=None,
                            top_k=None,
                            )

                input_length = inputs['input_ids'].shape[1]
                new_tokens = outputs[0][input_length:]
                raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                logger.info(f"Raw LLM output type: {raw_response}")

            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                return None
    
        else:
            logger.warning('prompt Constructor error')
            return None

        return raw_response.strip()


    def run_pipeline_from_result_dicts(self, force_rebuild: bool, documents_folder: str, model_id :str):

        """
        Run the RAG pipeline from pre-processed result dicts.

        Args:
            model_id: ID of the model to use for the LLM
            force_rebuild: Force rebuild even if cache is valid
            documents_folder: Path to documents folder for caching
        """


        if not force_rebuild:
            is_valid, reason = self.cache_manager.check_index_valid(
                documents_folder, self.config
            )
            is_valid = True #uncomment later
            if is_valid:
                logger.info(f"✓ Using cached index: {reason}")
                logger.info(f"\n✓ Using cached index (no changes detected)")

                self.processed_docs = self.cache_manager.load_processed_chunks()
                if self.processed_docs:
                        self.setup_retriever(self.processed_docs, force_rebuild)
                        logger.info(f" Loaded {len(self.processed_docs)} cached document chunks")
                        self.initialize_llm(model_id)
                        logger.info("✓ Index ready!")
                        return
                else:
                        logger.warning("Could not load cached chunks, rebuilding...")

            else:
                force_rebuild = True # to rebuild the qdrant indexing ,because the cache is not valid anymore
                logger.info(f"✗ Rebuilding documents: {reason}")

        else:
            logger.info("\n⟳ Force rebuild requested")


        logger.info("  Processing documents...")

        self.load_embedtokenizer()
        file_dir = Path("text_processing")
        file_dir.mkdir(exist_ok=True)
        processor = processDocument(
            tokenizer = self.embedtokenizer,
            chunk_size = self.config.CHUNK_SIZE_TOKENS,
            chunk_overlap = self.config.CHUNK_OVERLAP_TOKENS,
            output_dir=file_dir,
            pdf_folder=documents_folder,
            lang_model=lang_model,
            run_type=self.config.RUN_TYPE)
        end_chunks = processor.run() # process documents and creates chunks back
        for i, chunk in enumerate(end_chunks, 1):
            chunk['chunk_id'] = i

        #convert to doc objects
        end_chunks = [Document(page_content=chunk['content'], metadata={k:v for k,v in chunk.items() if k != 'content'}) for chunk in end_chunks]
        result_dir = Path(f"{file_dir}/Final")
        result_dir.mkdir(exist_ok=True)
        output_path = f'{result_dir}/{self.config.RUN_TYPE}rag_ready_chunks.json'
        with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(
                                [{"content": c.page_content, **c.metadata} for c in end_chunks],
                                f, indent=2, ensure_ascii=False
                            )

        self.unload_embedtokenizer() # unloading embedding tokenizer to free memory, since we dont need it anymore
        logger.info("Saving processed chunks...")
        self.cache_manager.save_processed_chunks(end_chunks)
        self.cache_manager.update_manifest(documents_folder, self.config)
        self.setup_retriever(end_chunks, force_rebuild) # this calls the qdrant system to index the documents

        self.initialize_llm(model_id)
        logger.info("✓ Index ready!")

def main():
    """Main function demonstrating the enhanced RAG system"""

    ##Configure system, you can adjust these settings as needed

    config = AdvancedRAGConfig(
        USE_HYBRID_SEARCH=True,
        USE_RERANKING=True,
        ENABLE_MONITORING=True,

        USE_8BIT= False,
        USE_4BIT= True,

    )

    # Path to your main technical documents folder
    documents_folder = "/home/anagha/bot_sample/app/backend/dataset"

    # # Initialize RAG system
    rag = AdvancedLangChainRAG(config)

    # #Run pipeline
    force_rebuild = False

    rag.run_pipeline_from_result_dicts(
        force_rebuild = force_rebuild,
        documents_folder = documents_folder,
        model_id = rag.config.LLM_MODEL_ID
    )

   
    # --------------- LIVE RAG SYSTEM INTERACTION -----------------


    logger.info("\n" + "="*80)
    logger.info("  ADVANCED RAG SYSTEM WITH DOCUMENT HANDLER SUPPORT")
    logger.info("="*80)

    while True:
        try:
            machine = [str(input(f" Optional - specify machine context (or press Enter to skip): ").strip())] # want it in list format
            
            user_q = input(f"\n Your question: ").strip()

            if not user_q:
                continue

            if user_q.lower() == 'quit':
                logger.info("\nGoodbye! See you next time.")
                break

            if user_q.lower() == 'clear':
                rag.clear_history()

                continue

            response = rag._rag_query(user_q, True, time.time(), machine)

            intent_display = response.get('intent', 'unknown')
            machine_display = response.get('machine_context', '')
            if response.get('awaiting_clarification'):
                intent_display += " (awaiting machine name)"

            logger.info(f"\n[Intent: {intent_display}] "
                  f"[Machine: {machine_display or 'None'}] "
                  f"[RAG: {'Yes' if response.get('used_rag', True) else 'No'}] "
                  f"[Time: {response['latency']:.2f}s]"
                  f"[Language: {response.get('query_language', 'unknown')}]")
            logger.info("-" * 80)
            logger.info(f"\n{response['answer']}")

            if response.get('sources'):
                logger.info(f"\n Sources ({len(response['sources'])} documents):")
                logger.info("-" * 80)
                for i, src in enumerate(response['sources'], 1):
                    metadata = src['metadata']
                    chunk_id = metadata.get('chunk_index', 'N/A')
                    score = metadata.get('rerank_score', 'N/A')
                    logger.info(f"Source {i} (Chunk ID: {chunk_id}):")
                    logger.info(f"Rerank Score: {score}")
                    logger.info('**************************************')
            rating = input("\nAnswer right or wrong? Press R for right and W for wrong (or Enter to skip): ").strip()
            if rag.monitor and rating:
                correctness = 'right' if rating.lower() == 'r' else 'wrong'
                rag.monitor.update_last_entry_correctness(correctness)
                logger.info(f"✓ Logged with correctness: {correctness}")

        except KeyboardInterrupt:
            logger.info("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            logger.info(f"Sorry, an error occurred: {e}")



