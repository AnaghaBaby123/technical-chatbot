import pymupdf4llm
import json
import re
import fasttext
import os
import spacy
from pathlib import Path
import logging
import unicodedata
from langchain_core.documents import Document
import pathlib
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class processDocument:
    
    def __init__(self, chunk_size, chunk_overlap, tokenizer, output_dir, pdf_folder, lang_model, run_type):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.pdf_folder = pdf_folder  
        self.lang_model = lang_model
        self.run_type = run_type
        
        logger.info(f'Chunk size {self.chunk_size}, Chunk overlap: {self.chunk_overlap}')

    def traverse_pdfs_as_dict(self, folder_path):
        """
        Returns a dictionary:
        {
            "dir_path": ["file1.pdf", "file2.pdf"],
            ...
        }
        """
        result = {}

        for root, dirs, files in os.walk(folder_path):
            pdfs = [
                os.path.join(root, f)
                for f in files
                if f.lower().endswith(".pdf")
            ]
            if pdfs:
                result[root] = pdfs

        return result

    def detect_language_fasttext(self, text: str, threshold: float = 0.6) -> dict:
            labels, probs = self.lang_model.predict(text)
            lang = labels[0].replace("__label__", "")
            conf = probs[0]
            conf = float(probs[0])

            return {
                        "language": lang if conf >= threshold else "en",
                        "confidence": conf
                    }
    
    def detect_language(self, chunks) :
                
                lang_info = None
                lang_detected = False
                for chunk in chunks:
                    if chunk["type"] == "text":
                        text = chunk["content"].strip()
                        #remove newlines and extra spaces for better detection
                        text = re.sub(r'\n', ' ', text)
                        text = re.sub(r' +', ' ', text)
                        if len(text) > 100:   # ignore very short text
                            print(text)
                            lang_info = self.detect_language_fasttext(text)['language']
                            lang_detected = True
                            return lang_info
                if not lang_detected: # if there are no text items, try tables Fallback method
                    for chunk in chunks:
                        if chunk["type"] == "table":
                            print("table")
                            
                            table_content = chunk["content"].strip().replace('\n',' ')
                            print(table_content)
                            lang_info = self.detect_language_fasttext(table_content)['language']
                            lang_detected = True
                            return lang_info
                return None
    
    def reclassify_item(self,chunks):
                """Reclassify misidentified headers"""
                
                for chunk in chunks:
                    if chunk["type"] != "section-header":
                        continue
                    text = chunk["content"].strip()
                    
                    # numbered header pattern , 1Title, 1 Title, etc.
                    #starts with digit and ?(allowed next is word or space) then 0 to more space and then letters
                    is_numbered_header = bool(re.match(r'^\d+(?=[A-Za-z\s])\s*\w+', text)) # assume numbered headers
                    #3.2.3Title, 3.2.3 Title) 
                    is_sub_header = (bool(re.match(r'^\d+\.\d+(?:\.\d+)*[.)]?\s*\w+', text)))
                                  
                                

                    sentence_endings = ('.', '!', '?',':',';')
                    if text and text[-1] in sentence_endings:
                        is_numbered_header = False

                    if is_sub_header:
                        text = '<subsec>' + text # to underline sub headers in markdown
                        return 'text', text
                
                    elif text and is_numbered_header :
                        return "section-header", text
                    else:
                        return "text", text  # Default not trust
    
    def combine_chunks_by_section(self, raw_chunks, file_name, machine_context, lang_info, access_control):
                            """
                            Combine all chunks belonging to the same section into a single chunk.
                            
                            """
                            
                            combined_chunks = []
                            current_section = None
                            section_data = None
                            
                            def finalize_section(section_data):
                                """Finalize and return a section chunk"""
                            
                                # Combine text content
                                combined_text = "\n".join(section_data["text_parts"]) # adding tables and texts inline
                            
                                return {
                                    "type": "section",
                                    "section": section_data["section"],  
                                    "content": combined_text.strip(),  
                                    "tables": section_data["tables"],
                                    #"images": section_data["images"],
                                    "page_start": section_data["page_start"],
                                    "page_end": section_data["page_end"],
                                    "file_name": file_name,
                                    "machine_context": machine_context,
                                    "language": lang_info,
                                    "access_control": access_control
                                }
                            
                            for chunk in raw_chunks:
                                
                                chunk_type = chunk.get("type", "")
                                if chunk_type == "section-header":
                                    section_value = chunk.get("content", "Untitled Section")
                                    
                                    section = re.sub(r'(\d)([A-Za-z])', r'\1 \2', section_value)
                                    

                                # Check if we're starting a new section
                                if section != current_section:
                                    # Finalize previous section
                                    if section_data is not None:
                                        finalized = finalize_section(section_data)
                                        if finalized and finalized["content"]:
                                            combined_chunks.append(finalized)
                                    
                                    # Start new section
                                    current_section = section
                                    section_data = {
                                        "section": section,
                                        "text_parts": [],
                                        "tables": [],
                                        "images": [],
                                        "page_start": chunk.get("page"),
                                        "page_end": chunk.get("page")
                                    }
                                
                                # Update page range
                                if chunk.get("page"):
                                    if section_data["page_start"] is None:
                                        section_data["page_start"] = chunk["page"]
                                    section_data["page_end"] = chunk["page"]
                                
                                # Add content based on type
                                if chunk_type in ["text"]:
                                    if chunk.get("content"):
                                        text = chunk["content"] #post processing because lines are written in pdf as separate text blocks
                                        # 1. Fix hyphenated line breaks
                                        content = re.sub(r'-\n', '', text)

                                        # 2. Preserve paragraph breaks (double newlines) by using a placeholder
                                        content = re.sub(r'\n{2,}', '<<PARA>>', content)

                                        # 3. Now collapse single newlines (mid-paragraph line wraps)
                                        content = re.sub(r'\n', ' ', content)

                                        # 4. Collapse extra spaces
                                        content = re.sub(r' +', ' ', content)

                                        # 5. Restore paragraph breaks
                                        content = content.replace('<<PARA>>', '\n\n')
                                        section_data["text_parts"].append(content)
                                
                                elif chunk_type == "table":
                                    content = f"\n ### Table\n{chunk.get('content','')}\n"
                                    #section_data["table_parts"].append(content)
                                    section_data["tables"].append({
                                        "content": chunk.get("content"),            
                                        "page_number": chunk.get("page"),
                                        "table_id" : chunk.get("table_id")
                                    })
                                
                                elif chunk["type"] == "image":
                                
                                    section_data["images"].append({
                                            "content": chunk.get("content",""),
                                            "page_number": chunk.get("page")
                                        })
                            
                            # Finalize the last section
                            if section_data is not None:
                                finalized = finalize_section(section_data)
                                if finalized and finalized["content"]:
                                    combined_chunks.append(finalized)
 
                            return combined_chunks
    
    def count_tokens(self, text: str) -> int:
                """
                Count tokens using the BGE-M3 tokenizer for accurate chunk sizing.
                Falls back to word splitting if tokenizer not available.  
                """
                if not text:
                    return 0
                
                if self.tokenizer is not None:
                    # Use actual tokenizer for accurate count
                    return len(self.tokenizer.encode(text, add_special_tokens=False))
                else:
                    # Fallback to word-based estimation (less accurate)
                    # Multiply by 1.3 as safety margin since subword tokens > words
                    logger.info("Using fallback word-based token count estimation.")
                    return int(len(text.split()) * 1.3)             
    

    def split_by_subsections(self, text: str) -> list[str]:
        """Split text by subsection headers."""
        parts = re.split(r'(^<subsec>)', text, flags=re.MULTILINE)
        
        result = []
        current = ""
        
        for part in parts: # split at subsec, but it comes as sep, so joining it to next part.
            if part == "<subsec>":
                if current:
                    result.append(current.strip())
                current = part
            else:
                current += part
        
        if current:
            result.append(current.strip())
        
        return result


    def split_by_sentences(self, text: str, language: str) -> list[str]:
        """
    
        spaCy/nltk if you need robust multilingual support:"""
        
        nlp_en = spacy.load("en_core_web_sm")
        nlp_de = spacy.load("de_core_news_sm")
        if language == "de":
            logger.info("Using German sentence splitter.")
            return [sent.text.strip() for sent in nlp_de(text).sents]
        else:
            logger.info("Using English sentence splitter.")
            return [sent.text.strip() for sent in nlp_en(text).sents]
        

    def get_overlap_text(self, text: str, overlap_tokens: int, base_metadata: dict) -> str:
        """
        Return the trailing `overlap_tokens` worth of text from `text`,
        snapped to a sentence boundary so the overlap is always coherent.
        Falls back to a raw token-based tail if no sentence boundary is found.
        """
        if not text or overlap_tokens <= 0:
            return ""

        sentences = self.split_by_sentences(text, base_metadata["language"])
        if not sentences:
            return ""

        overlap_parts = []
        token_count = 0

        for sentence in reversed(sentences):
            t = self.count_tokens(sentence)
            if token_count + t > overlap_tokens and overlap_parts:
                break
            overlap_parts.insert(0, sentence)
            token_count += t
            if token_count >= overlap_tokens:
                break

        return " ".join(overlap_parts).strip()


    def merge_units_into_chunks(
        self,
        units: list[str],
        overlap_prefix: str = "",
        base_metadata: dict | None = None,
        table_links: list = None,
        chunk_method: str = "semantic_merge",) -> list[dict]:
        """
        Greedy merge: accumulate `units` until chunk_size is reached,
        then flush and start a new chunk with an overlap prefix derived
        from the tail of the previous chunk (snapped to a sentence boundary).

        Never cuts a unit in half — the smallest atomic piece you pass in
        (sentence or paragraph) is always kept whole.

        Args:
            units:          Ordered list of atomic text units to merge.
            overlap_prefix: Optional text to prepend to the very first chunk.
            base_metadata:  Metadata dict to embed in every output chunk.
            chunk_method:   Value for the 'chunk_method' metadata field.

        Returns:
            List of chunk dicts ready for RAG ingestion.
        """
        if base_metadata is None:
            base_metadata = {}

        chunks = []
        current_parts: list[str] = []
        current_tokens = self.count_tokens(overlap_prefix) if overlap_prefix else 0

        def flush(parts: list[str], prefix: str) -> str:
            """Build chunk text and append to chunks list; return new overlap prefix."""
            body = "\n".join(parts)
            full_text = (prefix + "\n" + body).strip() if prefix else body.strip()

            #check if there are any table tags in the section, if so ass it into metadata for further ref
            # and then remove the tags in content so embedding is not problem
            
            table_links = []   
            if "<table>" in full_text:
                table_links.extend(re.findall(r"<table>(.*?)</table>", full_text, re.DOTALL))
                full_text = re.sub(r"<table>.*?</table>", " ", full_text, flags=re.DOTALL)
                logger.info(f"FLUSH :table links are \n {table_links}")

            chunks.append({
                **base_metadata,
                "content": full_text,
                "token_count": self.count_tokens(full_text),
                "chunk_type": chunk_method,
                "table_links" : table_links,
                "has_overlap": bool(prefix),
            })
            # Derive overlap from the raw body (not including the incoming prefix)
            # so that overlap doesn't compound across chunks.
            if "<table>" in body:
                body = re.sub(r"<table>.*?</table>", " ", body, flags=re.DOTALL)
                
            return self.get_overlap_text(body, self.chunk_overlap, base_metadata)

        for unit in units:
            unit_tokens = self.count_tokens(unit)

            # Edge case: a single unit is itself larger than chunk_size.
            # We can't split it here,
            # so we flush whatever we have, then emit this unit alone.
            if unit_tokens > self.chunk_size:
                if current_parts:
                    overlap_prefix = flush(current_parts, overlap_prefix)
                    current_parts = []
                    current_tokens = self.count_tokens(overlap_prefix)

                # Emit oversized unit as its own chunk with a warning.
                logger.warning(
                    f"Unit exceeds chunk_size ({unit_tokens} > {self.chunk_size}). "
                    "Emitting as oversized chunk. Consider splitting at a finer level."
                )
                full_text = (overlap_prefix + "\n" + unit).strip() if overlap_prefix else unit.strip()
               
                if "<table>" in full_text:
                    table_links.extend(re.findall(r"<table>(.*?)</table>", full_text, re.DOTALL))
                    full_text = re.sub(r"<table>.*?</table>", " ", full_text, flags=re.DOTALL)
                    logger.info(f"OVER: table links are \n {table_links}")
                chunks.append({
                    **base_metadata,
                    "content": full_text,
                    "token_count": self.count_tokens(full_text),
                    "chunk_type": chunk_method + "_oversized",
                    "table_links" : table_links,
                    "has_overlap": bool(overlap_prefix),
                })
                if "<table>" in unit: #removing table tags for overlap
                    unit = re.sub(r"<table>.*?</table>", " ", unit, flags=re.DOTALL)
                overlap_prefix = self.get_overlap_text(unit, self.chunk_overlap, base_metadata)
                current_tokens = self.count_tokens(overlap_prefix)
                continue

            # Normal case: adding this unit would exceed chunk_size — flush first.
            if current_tokens + unit_tokens > self.chunk_size and current_parts:
                overlap_prefix = flush(current_parts, overlap_prefix)
                current_parts = []
                current_tokens = self.count_tokens(overlap_prefix)

            current_parts.append(unit)
            current_tokens += unit_tokens

        # Flush the final accumulated parts.
        if current_parts:
            flush(current_parts, overlap_prefix)

        return chunks

    def process_tables_as_chunks(self, tables: list[dict], base_metadata: dict) -> list[dict]:
        """
        Process tables as separate chunks with section metadata but no overlap.
        They are not merged with text to avoid breaking table structure and meaning.
        If a table is bigger than chunk_size, it will be emitted as its own chunk with a chunk_method of "table_oversized" and a warning will be logged.
        """
        table_chunks = []
        for table in tables:
            content = table.get("content", "")
            token_count = self.count_tokens(content)
            if token_count > self.chunk_size:
                logger.warning(
                    f"Table exceeds chunk_size ({token_count} > {self.chunk_size}). "
                    "Emitting as oversized chunk. Consider summarizing or splitting the table."
                )
                chunk_method = "table_oversized"
            else:
                chunk_method = "table"

            table_chunks.append({
                **base_metadata,
                "content": content,
                "table_id" : table.get("table_id"),
                "token_count": token_count,
                "chunk_method": chunk_method,
                "table_pagenumber": table.get("page_number"),
                "has_overlap": False,
            })

        return table_chunks

    def check_all_subsections(self, subsections):
        
        """
        Check all subsections:
        if its less than chunk overlap, combine with the next one.
        This is to avoid the duplicates when one subsection is less than chunk overlaped, it merges with another chunk with that same content
        Edge case: if last chunk is less than chunk overlap:: it is kept as it
        """
        i = 0
        while i < len(subsections) - 1:
                if self.count_tokens(subsections[i]) <= self.chunk_overlap:
                    subsections[i + 1] = subsections[i] + subsections[i + 1]
                    del subsections[i]
                else:
                    i += 1
        
        return subsections

    def extract_heading(self,text):
        
        matches = re.findall(r"<subsec>\s*(\d+(?:\.\d+)*)([^\n]+)", text)
        headings = []
        headings = [f"{num} {title.strip()}" for num, title in matches]
        
        if not headings:
                headings = ['Untitled Subsection']
        return headings
    
    def chunk_section_for_rag(self, section_chunk: dict) -> list[dict]:
        """
        Chunk a section for RAG ingestion using semantic boundaries.

        Strategy (in order of preference):
        1. Section fits in chunk_size            → return as-is
        2. Section has ## subsections            → merge subsections greedily;
                                                    sentence-split any subsection
                                                    that is itself too large
        4. Fallback                              → sentence-split the whole section

        Tables:
        as seperate chunks with section metadata but no overlap, and a chunk_method of "table". They are not merged with text to avoid breaking table structure and meaning.
        if table is bigger than chunk_size, it will be emitted as its own chunk with a chunk_method of "table_oversized" and a warning will be logged.

        Overlap is always derived by snapping to the nearest sentence boundary,
        so chunks never start or end mid-sentence.

        Also getting table links to each correspoind chunks.
        for sentence split: not possible (need better idea)

        Args:
            section_chunk: Combined section chunk dict with 'content' and metadata.
            

        Returns:
            List of RAG-ready chunk dicts with metadata.
        """
        table_links = []
        content = section_chunk.get("content", "")
        # handling edge case where table tags are there in section title
        section_title = section_chunk.get("section"," ")
        if "<table>" in section_title:
                logger.info("table links found in section title")
                table_links = re.findall(r"<table>(.*?)</table>", section_title, re.DOTALL)
                section_title = re.sub(r"<table>.*?</table>", " ", section_title , flags=re.DOTALL)
                logger.info(f"table links are \n {table_links}")
        logger.info(f"section title {section_title}")
        content = section_title + '\n' + content

        tables = section_chunk.get("tables", [])
        logger.info(f"Section {section_chunk.get('section')} got {len(tables)} tables")
   
        token_count = self.count_tokens(content)
        logger.info(f"Section '{section_chunk.get('section')}' has {token_count} tokens.")

        base_metadata = {
            "section_heading": section_title,
            
            "file_name":       section_chunk.get("file_name"),
            "machine_context": section_chunk.get("machine_context"),
            "language":        section_chunk.get("language"),
            "access_control":  section_chunk.get("access_control"),
            #"images":          section_chunk.get("images", []),
            "section_tables":    section_chunk.get("tables", []),
            "page_start":      section_chunk.get("page_start"),
            "page_end":        section_chunk.get("page_end"),
        }
        table_metadata = {
            "section_heading":  section_chunk.get("section"),
            "file_name":       section_chunk.get("file_name"),
            "machine_context": section_chunk.get("machine_context"),
            "language":        section_chunk.get("language"),
            "access_control":  section_chunk.get("access_control"),
        }
        table_chunks = self.process_tables_as_chunks(tables , table_metadata)

        # ------------------------------------------------------------------ #
        # Case 1: section fits within chunk_size — return as-is               #
        # ------------------------------------------------------------------ #
        if token_count <= self.chunk_size:
            rag_chunks = [] 
            #check if there are any table tags in the section, if so ass it into metadata for further ref
            # and then remove the tags in content so embedding is not problem
            if "<table>" in content:
                table_links.extend(re.findall(r"<table>(.*?)</table>", content, re.DOTALL))
                content = re.sub(r"<table>.*?</table>", " ", content, flags=re.DOTALL)
                logger.info(f"table links Case 1 \n {table_links}")
            content = re.sub(r'(<subsec>\d\.\d)([A-Za-z])', r'\1 \2', content) 
            content = content.replace('<subsec>', '\n')
            content = self.clean_text_for_chunking(content)
            
            
            rag_chunks.append({
                **base_metadata,
                "content": content,
                "token_count": token_count,
                "table_links" : table_links,
                "chunk_method": "section_fit",
                "has_overlap": False,
            })
            rag_chunks.extend(table_chunks)
            return rag_chunks

        # ------------------------------------------------------------------ #
        # Case 2: split by ## subsections, merge greedily                     #
        # ------------------------------------------------------------------ #
        subsections = self.split_by_subsections(content)
        logger.info(f'before checkings len: {len(subsections)}')
        subsections = self.check_all_subsections(subsections)
        
        if len(subsections) > 1:
            logger.info(f"Splitting by subsections ({len(subsections)} found).")
            rag_chunks: list[dict] = []
            overlap_prefix = ""

            for subsec in subsections:
          
                #replace the  identifier used to create subsections, replaced it with newline
                subsection_headers = self.extract_heading(subsec)
                subsection_headers = ', '.join(subsection_headers) 
                if "<table>" in subsection_headers:
                    logger.info("table links found in susection_headers")
                    subsection_headers = re.sub(r"<table>.*?</table>", "", subsection_headers , flags=re.DOTALL)
                  
                
                base_metadata['sub_section_heading'] = subsection_headers

                logger.info(f'subsection headers \n {subsection_headers}')

                subsec = subsec.replace('<subsec>', '\n')
                subsec = re.sub(r'(\d\.\d)([A-Za-z])', r'\1 \2', subsec) # to get space between numerals.
               
                subsec = self.clean_text_for_chunking(subsec)
              
                subsec_tokens = self.count_tokens(subsec)
              
                
                if subsec_tokens <= self.chunk_size:
                    # This subsection fits — hand it to merge_units_into_chunks
                    # as a single-element list so overlap logic is consistent.
                    new_chunks = self.merge_units_into_chunks(
                        units=[subsec],
                        overlap_prefix=overlap_prefix,
                        base_metadata=base_metadata,
                        chunk_method="subsection_merge",
                        table_links = table_links
                    )
                    rag_chunks.extend(new_chunks)
                    if new_chunks:
                        overlap_prefix = self.get_overlap_text(
                            new_chunks[-1]["content"], self.chunk_overlap, base_metadata
                        )
                else:
                    logger.info(
                        f"Subsection too large ({subsec_tokens} tokens), "
                        "falling back to sentence split."
                    )
                    if "<table>" in subsec: #skipping table links logic in sentence split as table tags are not split accordingly, need better ways
                        #table_links.extend(re.findall(r"<table>(.*?)</table>", subsec, re.DOTALL))
                        subsec = re.sub(r"<table>.*?</table>", " ", subsec, flags=re.DOTALL)
                        table_links = []
                        #logger.info(f"table links Case 1 \n {table_links}")
                    sentences = self.split_by_sentences(subsec, base_metadata["language"])
                    logger.info(f"Split subsection into {len(sentences)} sentences for merging.")
                    
                    new_chunks = self.merge_units_into_chunks(
                            units=sentences,
                            overlap_prefix=overlap_prefix,
                            base_metadata=base_metadata,
                            chunk_method="subsection_sentence_merge",
                            table_links = table_links,
                        )

                    rag_chunks.extend(new_chunks)
                    if new_chunks:
                        overlap_prefix = self.get_overlap_text(
                            new_chunks[-1]["content"], self.chunk_overlap, base_metadata
                        )
            rag_chunks.extend(table_chunks) # add tables at the end of the section's chunks
            return rag_chunks

      
        logger.info("No subsections found")

        # ------------------------------------------------------------------ #
        # Case 4: fallback — sentence-split the whole section                 #
        # ------------------------------------------------------------------ #
        logger.info("splitting by sentences.")
        content = re.sub(r'(<subsec>\d\.\d)([A-Za-z])', r'\1 \2', content) 
        content = content.replace('<subsec>', '\n')
        if "<table>" in content:
            #table_links.extend(re.findall(r"<table>(.*?)</table>", subsec, re.DOTALL))
            subsec = re.sub(r"<table>.*?</table>", " ", content, flags=re.DOTALL)
            table_links = []
            #logger.info(f"table links Case 1 \n {table_links}")
        content = self.clean_text_for_chunking(content)
        sentences = self.split_by_sentences(content, base_metadata["language"])
        
        
        rag_chunks = self.merge_units_into_chunks(
            units=sentences,
            base_metadata=base_metadata,
            chunk_method="sentence_merge",
            table_links = table_links
        )
        rag_chunks.extend(table_chunks) # add tables at the end of the section's chunks

        return rag_chunks

        
    
    def process_all_sections_for_rag(self, final_chunks: list) -> list:
            """
            Process all section chunks and create RAG-ready chunks.
            
            Args:
                final_chunks: List of combined section chunks
                
                
            Returns:
                List of RAG-ready chunks with assigned IDs
            """
            all_rag_chunks = []
            
            for section_chunk in final_chunks:
                logger.info(f"\nProcessing section for RAG: {section_chunk.get('section')}\n")
                rag_chunks = self.chunk_section_for_rag(section_chunk)
                all_rag_chunks.extend(rag_chunks)

            
            return all_rag_chunks

    


    def clean_text_for_chunking(self, text: str) -> str:

        # 1. Strip control characters (except \n and \t which we handle later)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # 2. Remove Private Use Area characters (e.g. \uf0fe Wingdings checkmarks)
        text = re.sub(r'[\uE000-\uF8FF]', '', text)          # BMP PUA
        text = re.sub(r'[\U000F0000-\U000FFFFF]', '', text)  # Supplementary PUA-A
        text = re.sub(r'[\U00100000-\U0010FFFF]', '', text)  # Supplementary PUA-B

        # 3. Remove zero-width and invisible characters
        text = re.sub(r'[\u200B-\u200D\uFEFF\u00AD]', '', text)

        # 4. Remove replacement character
        text = re.sub(r'\uFFFD', '', text)

        # 5. Normalize unicode (NFC = composed form, consistent representation)
        text = unicodedata.normalize('NFC', text)

        # 6. Normalize whitespace
        text = text.replace('\t', ' ')           # tabs → spaces
        text = re.sub(r'[ ]+', ' ', text)        # collapse multiple spaces
        text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 consecutive newlines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = text.strip()

        return text

    def process_singlefile(self, file_path, machine_context = None, access_control = None):

            json_string = pymupdf4llm.to_json(file_path, header=False, footer=False)
            data = json.loads(json_string)
            stemname = pathlib.Path(file_path).stem
            raw_dir = Path(f"{self.output_dir}/Raw_outputs")
            raw_dir.mkdir(exist_ok=True)
            output_path = raw_dir / f"{self.run_type}_raw_{stemname}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            #----- Extracting chunks from raw data
            chunks = []
            unknowns = []       
            filename = data.get("filename", "unknown.pdf")
            file_name = pathlib.Path(filename).stem
            
            logger.info(f"Processing file: {file_name}")
            blockTypeslist = []
            table_no = 1
            for page in data.get("pages", []):
                if not page:
                    logger.warning(f"Empty page data in file {file_name}")
                page_number = page.get("page_number")
                
                for block in page.get("boxes", []):
                    if block is None:
                        logger.warning(f"Empty block in page {page_number} of file {file_name}")
                        break
                    block_type = block.get("boxclass")
                    
                    bbox = block.get("x0"), block.get("y0"), block.get("x1"), block.get("y1")
                    
                    # Text block (type 0)
                    if block_type.lower() == "section-header":
                        
                        text = ""
                        for line in block.get("textlines", []):
                            for span in line.get("spans", []):
                                text += span.get("text", "")
                            text += "\n"
                        text = text.strip()
                        if text:
                            blockTypeslist.append(block_type)
                            chunks.append({
                                "type": "section-header",
                                "content": text,
                                "bbox": bbox,
                                "page": page_number,
                                "file_name": file_name
                            })

                    # Image block (type 1)
                    elif block_type.lower() == "picture":
                        blockTypeslist.append(block_type)
                        chunks.append({
                            "type": "image",
                            "content": block.get("image", None),  # base64 or None
                            "bbox": bbox,
                            "page": page_number,
                            "file_name": file_name
                        })
                        text = ""
                        for line in block.get("textlines", []):
                            for span in line.get("spans", []):
                                text += span.get("text", "")
                            text += "\n"
                        text = text.strip()
                        if text:
                            blockTypeslist.append("text")
                            chunks.append({
                                "type": "text",
                                "content": text,
                                "bbox": bbox,
                                "page": page_number,
                                "file_name": file_name
                            })    

                    elif block_type.lower() == "text":
                        
                        text = ""
                        for line in block.get("textlines", []):
                            for span in line.get("spans", []):
                                text += span.get("text", "")
                            text += "\n"
                        text = text.strip()
                        if text:
                            blockTypeslist.append(block_type)
                            chunks.append({
                                "type": "text",
                                "content": text,
                                "bbox": bbox,
                                "page": page_number,
                                "file_name": file_name
                            })

                    # Table block
                    elif block_type.lower() == "table" :
                        
                        table = block.get("table", "")
                        markdown_table = table.get("markdown", "")
                        if markdown_table:
                            blockTypeslist.append(block_type)
                            chunks.append({
                                "type": "table",
                                "table_id" : f"{file_name}_{table_no}",
                                "content": markdown_table,
                                "bbox": bbox,
                                "page": page_number,
                                "file_name": file_name
                                })
                            table_no+=1
                    
                    elif block_type.lower() == "page-header" or block_type.lower() == "page-footer" :
                        continue
                    
                    elif block_type.lower() == "list-item": 
                        
                        text = ""
                        for line in block.get("textlines", []):
                            for span in line.get("spans", []):
                                text += span.get("text", "")
                            text += "\n"
                        text = text.strip()
                        if text:
                            blockTypeslist.append("text")
                            chunks.append({
                                "type": "text",  # reclassify list items as text for better processing
                                "content": text,
                                "bbox": bbox,
                                "page": page_number,
                                "file_name": file_name
                            })
                
                        

                    # Fallback for any other type
                    else:
                        logger.warning(f"Unknown block type '{block_type}' in page {page_number} of file {file_name}")
                        unknowns.append({
                            "type": f"unknown_{block_type}",
                            "content": block,
                            "bbox": bbox,
                            "page": page_number,
                            "file_name": file_name
                        })

            
            

            #-------------reclassification of section headers as text for better processing  
            
            for chunk in chunks:
                if chunk["type"] == "section-header":
                    # reclassifying section headers as text for better processing
                    reclassified_type, reclassified_content = self.reclassify_item([chunk])
                    chunk["type"] = reclassified_type
                    chunk["content"] = reclassified_content

            #put a placeholder for tables in before block

            table_textmap = {}
            last_text_index = None
            
            for i, val in enumerate(blockTypeslist):
                if val not in ("table", "picture"):
                    last_text_index = i
                elif val == "table":
                    table_textmap[i] = last_text_index if last_text_index is not None else 0
            for key, val in table_textmap.items():
                text_chunk = chunks[val]
                table_chunk = chunks[key]
                text_chunk["content"] = text_chunk.get("content") + "<table>" + table_chunk.get("table_id") + "</table>"



            #---------combining chunks for sections

            # put a placeholder for first chunk if its not a section header to maintain structure
            if chunks and chunks[0]["type"] != "section-header":
                chunks.insert(0, {
                    "type": "section-header",
                    "content": "Untitled Section",
                    "bbox": None,
                    "page": None,
                    "file_name": file_name
                })
            
            lang_info = self.detect_language(chunks)

            
            final_chunks = self.combine_chunks_by_section(raw_chunks = chunks, file_name = file_name, machine_context = machine_context, lang_info = lang_info, access_control = access_control)
           


            #--------combining chunks for RAG-ready format

            rag_ready_chunks = self.process_all_sections_for_rag(final_chunks)
           

            return rag_ready_chunks

            
        
    def run(self):

            if self.pdf_folder.lower().endswith(".pdf"): # just a file is passed
                    end_chunks = self.process_singlefile(self.pdf_folder, machine_context=None, access_control=None)
                    
            else:
                files = self.traverse_pdfs_as_dict(self.pdf_folder)
                logger.info(f"Found {sum(len(v) for v in files.values())} PDF files in {len(files)} folders.\n")
                end_chunks = []
                for folder, file_list in files.items():
                    logger.info(f"\nProcessing folder: {folder}\n")
                    folder_base = folder.split('/')[-1]
                    machine_context = ' '.join(folder_base.split()[:-1]).lower()  
                    access_control = folder.split()[-1]  
                    logger.info(f"Machine context: {machine_context}, Access control: {access_control}\n")
                    for file in file_list: 
                        chunks = self.process_singlefile(file, machine_context, access_control)
                        end_chunks.extend(chunks)

            
            
                
            return end_chunks






    