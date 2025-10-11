import json
from typing import List, Optional, Dict, Union, Tuple
from pathlib import Path
import logging
from collections import deque

from mistralai import TextChunk

from schemas.indexing.indexing_models import PaperChunk, ChunkMetadata
from schemas.parser.parser_models import ParsedPaper, PaperSection

logger = logging.getLogger(__name__)

class HeadingChunk:
    """ Represents a chunk of text associated with a heading in a document. """

    def __init__(self, max_chunk_size: int = 350, min_chunk_size: int = 100, overlap: int = 0):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def chunk_paper(self, paper_data: ParsedPaper) -> List[PaperChunk]:
        """ Chunk the text under the heading into smaller pieces if it exceeds max_chunk_size. """

        chunks: List[PaperChunk] = []
        arxiv_id = paper_data.metadata.arxiv_id
        
        try: 
            section_chunks = self._chunk_by_section(paper_data.content.sections)
            if section_chunks:
                # Set arxiv_id for each chunk
                for chunk in section_chunks:
                    chunk.arxiv_id = arxiv_id
                chunks.extend(section_chunks)
                logger.info(f"Created {len(section_chunks)} section-based chunks for {arxiv_id}")
        except Exception as e:
            logger.error(f"Error chunking sections for {arxiv_id}: {e}")
            section_chunks = []
        
        try: 
            table_chunks = self._chunk_by_table(paper_data.content.tables)
            if table_chunks:
                # Set arxiv_id for each chunk
                for chunk in table_chunks:
                    chunk.arxiv_id = arxiv_id
                chunks.extend(table_chunks)
                logger.info(f"Created {len(table_chunks)} table-based chunks for {arxiv_id}")
        except Exception as e:
            logger.error(f"Error chunking tables for {arxiv_id}: {e}")
            table_chunks = []

        return chunks
    
    def _chunk_by_section(self, sections: List[PaperSection]) -> List[PaperChunk]:
        """ Chunk text by section with header-based chunking and smart merging.
        
        Algorithm:
        1. Group sections by their headers (maintain section boundaries)
        2. Within each header group, apply smart merging for small chunks
        3. Split large chunks while preserving readability
        
        Rules:
        - Each chunk: 100 words <= chunk <= 350 words (preferred)
        - Merging ONLY happens within the same section header
        - Smart merge: prefer merging with smaller neighbor
        - If cannot merge (would exceed max or no valid neighbors), keep as-is
        - Each chunk includes the section header
        """
        chunks: List[PaperChunk] = []
        
        if not sections:
            return chunks

        # Build initial section list with headers and text paired
        section_list = self._build_section_list(sections)
        
        if not section_list:
            logger.info("No valid sections after filtering.")
            return chunks
        
        # Group sections by header to maintain section boundaries
        grouped_sections = self._group_by_header(section_list)
        
        # Process each header group independently with smart merging
        processed_sections = []
        for header_group in grouped_sections:
            merged_group = self._smart_merge_within_group(header_group)
            processed_sections.extend(merged_group)

        # Split large sections and create final chunks
        chunks = self._create_chunks_from_sections(processed_sections)
        
        return chunks
    
    def _build_section_list(self, sections: List[PaperSection]) -> List[Dict]:
        """Build a list of sections with paired headers and text.
        
        Special handling:
        - Merges consecutive list_items to the previous text chunk
        - Skips list_items under References section
        - Filters out sections with less than 20 words
        
        Returns:
            List of dicts with 'header', 'text', 'prov', and 'word_count' keys
        """
        section_list = []
        current_header = ""
        pending_list_items = []  # Buffer for list_items to merge with previous text
        
        for i, section in enumerate(sections):
            if section.label == "section_header":
                current_header = section.content.strip()
                # Flush any pending list items to the last section before changing header
                if pending_list_items and section_list:
                    self._merge_list_items_to_last_section(section_list, pending_list_items)
                    pending_list_items = []
                    
            elif section.label == "text":
                # First, merge any pending list_items from before
                if pending_list_items and section_list:
                    self._merge_list_items_to_last_section(section_list, pending_list_items)
                    pending_list_items = []
                
                text_content = section.content.strip()
                word_count = len(text_content.split())

                # Skip sections with less than 20 words
                if len(text_content.split()) < 20:
                    logger.debug(f"Skipping section '{current_header}' with only {word_count} words (< 20)")
                    continue
                
                section_list.append({
                    'header': current_header,
                    'text': text_content,
                    'prov': [section.prov] if section.prov else [],
                    'word_count': word_count
                })
                
            elif section.label == "list_item":
                # Skip list_items under References section
                if self._is_reference_header(current_header):
                    logger.debug(f"Skipping list_item under References section: {current_header}")
                    continue
                
                # Buffer this list_item to merge with previous text
                text_content = section.content.strip()
                
                pending_list_items.append({
                    'text': text_content,
                    'prov': section.prov,
                    'word_count': word_count
                })
        
        # Merge any remaining pending list items at the end
        if pending_list_items and section_list:
            self._merge_list_items_to_last_section(section_list, pending_list_items)
        
        return section_list
    
    def _is_reference_header(self, header: str) -> bool:
        """Check if header is a References section."""
        if not header:
            return False
        
        reference_keywords = [
            'references', 'reference', 'bibliography', 'works cited'
        ]
        header_lower = header.lower().strip()
        
        return any(keyword == header_lower or header_lower.startswith(keyword) 
                   for keyword in reference_keywords)
    
    def _merge_list_items_to_last_section(self, section_list: List[Dict], list_items: List[Dict]) -> None:
        """Merge buffered list_items to the last section in section_list. This modifies section_list in place. Allows exceeding max_chunk_size."""

        if not section_list or not list_items:
            return
        
        last_section = section_list[-1]
        
        # Combine all list items into one text block
        list_items_text = "\n".join(item['text'] for item in list_items)
        list_items_word_count = sum(item['word_count'] for item in list_items)
        
        # Merge to last section
        last_section['text'] = last_section['text'] + "\n" + list_items_text
        last_section['word_count'] += list_items_word_count
        
        # Merge provenance
        for item in list_items:
            if item['prov']:
                last_section['prov'].append(item['prov'])
        
        logger.debug(f"Merged {len(list_items)} list_items ({list_items_word_count} words) to last section. "
                    f"New word count: {last_section['word_count']}")

    def _group_by_header(self, section_list: List[Dict]) -> List[List[Dict]]:
        """Group sections by their header to maintain section boundaries.
        
        Returns:
            List of groups, where each group contains sections with the same header
        """
        if not section_list:
            return []
        
        grouped = []
        current_group = []
        current_header = None
        
        for section in section_list:
            section_header = section['header']
            
            if current_header is None:
                # First section
                current_header = section_header
                current_group = [section]
            elif section_header == current_header:
                # Same header, add to current group
                current_group.append(section)
            else:
                # Different header, finalize current group and start new one
                if current_group:
                    grouped.append(current_group)
                current_header = section_header
                current_group = [section]
        
        # Don't forget the last group
        if current_group:
            grouped.append(current_group)
        
        logger.info(f"Grouped {len(section_list)} sections into {len(grouped)} header groups")
        return grouped
    
    def _smart_merge_within_group(self, group: List[Dict]) -> List[Dict]:
        """Smart merge sections within the same header group.
        
        Algorithm:
        - Use deque for O(1) operations
        - For each small section (< min_chunk_size):
          1. Check if merging with previous would be valid (not exceed max)
          2. Check if merging with next would be valid (not exceed max)
          3. Choose the smaller neighbor to merge with
          4. If neither is valid, keep section as-is
        
        Returns:
            List of merged sections within this group
        """
        if not group:
            return []
        
        if len(group) == 1:
            # Single section in group, return as-is
            return group
        
        # Convert to list for index-based access
        sections = list(group)
        n = len(sections)
        merged_flags = [False] * n  # Track which sections have been merged
        
        # First pass: identify and merge small sections
        i = 0
        while i < n:
            if merged_flags[i]:
                i += 1
                continue
            
            current = sections[i]
            
            if current['word_count'] >= self.min_chunk_size:
                # Section is large enough, skip
                i += 1
                continue
            
            # Section is too small, find best merge candidate
            prev_idx = self._find_previous_unmerged(i, merged_flags)
            next_idx = self._find_next_unmerged(i, merged_flags)
            
            # Calculate merge costs
            can_merge_prev = False
            can_merge_next = False
            prev_cost = float('inf')
            next_cost = float('inf')
            
            if prev_idx is not None:
                prev_section = sections[prev_idx]
                merged_count = current['word_count'] + prev_section['word_count']
                if merged_count <= self.max_chunk_size:
                    can_merge_prev = True
                    prev_cost = prev_section['word_count']
            
            if next_idx is not None:
                next_section = sections[next_idx]
                merged_count = current['word_count'] + next_section['word_count']
                if merged_count <= self.max_chunk_size:
                    can_merge_next = True
                    next_cost = next_section['word_count']
            
            # Decide merge direction based on smart strategy
            if can_merge_prev and can_merge_next:
                # Both directions valid, choose smaller neighbor
                if prev_cost <= next_cost:
                    # Merge with previous
                    sections[prev_idx] = self._merge_two_sections(sections[prev_idx], current)
                    merged_flags[i] = True
                    logger.debug(f"Merged section {i} with previous {prev_idx} (smaller: {prev_cost} vs {next_cost})")
                else:
                    # Merge with next
                    sections[next_idx] = self._merge_two_sections(current, sections[next_idx])
                    merged_flags[i] = True
                    logger.debug(f"Merged section {i} with next {next_idx} (smaller: {next_cost} vs {prev_cost})")
            elif can_merge_prev:
                # Only previous is valid
                sections[prev_idx] = self._merge_two_sections(sections[prev_idx], current)
                merged_flags[i] = True
                logger.debug(f"Merged section {i} with previous {prev_idx} (only option)")
            elif can_merge_next:
                # Only next is valid
                sections[next_idx] = self._merge_two_sections(current, sections[next_idx])
                merged_flags[i] = True
                logger.debug(f"Merged section {i} with next {next_idx} (only option)")
            else:
                # Cannot merge, keep as-is
                logger.debug(f"Cannot merge section {i} ({current['word_count']} words) - keeping as-is")
            
            i += 1
        
        # Build result from unmerged sections
        result = [sections[i] for i in range(n) if not merged_flags[i]]
        
        logger.info(f"Group with {len(group)} sections merged to {len(result)} sections")
        return result
    
    def _find_previous_unmerged(self, current_idx: int, merged_flags: List[bool]) -> Optional[int]:
        """Find the nearest previous section that hasn't been merged."""
        for i in range(current_idx - 1, -1, -1):
            if not merged_flags[i]:
                return i
        return None
    
    def _find_next_unmerged(self, current_idx: int, merged_flags: List[bool]) -> Optional[int]:
        """Find the nearest next section that hasn't been merged."""
        for i in range(current_idx + 1, len(merged_flags)):
            if not merged_flags[i]:
                return i
        return None
    
    def _merge_two_sections(self, section1: Dict, section2: Dict) -> Dict:
        """Merge two sections into one. Combines text, preserves both headers, and updates metadata."""
        # Combine headers with proper formatting
        combined_header = section1['header']
        if section2['header'] and section2['header'] != section1['header']:
            combined_header += f"\n{section2['header']}"
        
        # Combine text content
        combined_text = section1['text'] + "\n" + section2['text']
        combined_word_count = len(combined_text.split())
        
        # Combine provenance information (now handling lists)
        combined_prov = []
        if section1.get('prov'):
            if isinstance(section1['prov'], list):
                combined_prov.extend(section1['prov'])
            else:
                combined_prov.append(section1['prov'])
        if section2.get('prov'):
            if isinstance(section2['prov'], list):
                combined_prov.extend(section2['prov'])
            else:
                combined_prov.append(section2['prov'])
        
        return {
            'header': combined_header,
            'text': combined_text,
            'prov': combined_prov if combined_prov else [],
            'word_count': combined_word_count
        }
    
    def _create_chunks_from_sections(self, sections: List[Dict]) -> List[PaperChunk]:
        """Create final chunks from processed sections, splitting if needed.
        
        If a section exceeds max_chunk_size, split it into multiple chunks.
        """
        chunks = []
        
        for idx, section in enumerate(sections):
            word_count = section['word_count']
            
            if word_count <= self.max_chunk_size:
                # Section fits in one chunk
                full_text = self._format_chunk_text(section['header'], section['text'])
                chunk = self._create_chunk(
                    text=full_text,
                    chunk_id=f"section_{idx+1}",
                    section_heading=section['header'],
                    prov=section['prov']
                )
                chunks.append(chunk)
            else:
                # Section is too large, split it
                sub_chunks = self._split_large_section(section, idx)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_large_section(self, section: Dict, section_idx: int) -> List[PaperChunk]:
        """Split a large section into multiple chunks.
        
        Algorithm:
        - Split by sentences to maintain readability
        - Target chunks around 300 words (middle of min-max range)
        - Keep header in each sub-chunk for context
        """
        chunks = []
        header = section['header']
        text = section['text']
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        current_chunk_sentences = []
        current_word_count = 0
        sub_chunk_idx = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence would exceed max size
            if current_word_count + sentence_words > self.max_chunk_size and current_chunk_sentences:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk_sentences)
                full_text = self._format_chunk_text(header, chunk_text)
                
                chunk = self._create_chunk(
                    text=full_text,
                    chunk_id=f"section_{section_idx+1}_part_{sub_chunk_idx+1}",
                    section_heading=header,
                    prov=section['prov']
                )
                chunks.append(chunk)
                
                # Reset for next chunk
                current_chunk_sentences = [sentence]
                current_word_count = sentence_words
                sub_chunk_idx += 1
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_words
        
        # Get last chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            full_text = self._format_chunk_text(header, chunk_text)
            
            chunk = self._create_chunk(
                text=full_text,
                chunk_id=f"section_{section_idx+1}_part_{sub_chunk_idx+1}",
                section_heading=header,
                prov=section['prov']
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter. Splits on common sentence endings while preserving abbreviations."""
        import re

        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _format_chunk_text(self, header: str, text: str) -> str:
        """Format chunk text with header."""
        if header:
            return f"{header}\n{text}"
        return text
    
    def _create_chunk(self, text: str, chunk_id: str, section_heading: str, prov) -> PaperChunk:
        """Create a PaperChunk object."""
        word_count = len(text.split())
        
        # Flatten prov if it's a nested list
        flattened_prov = self._flatten_prov(prov)
        
        return PaperChunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_id=chunk_id,
                start_char=0,
                end_char=len(text),
                word_count=word_count,
                section_heading=section_heading,
                prov=flattened_prov
            ),
            arxiv_id=""  # to be set later by caller
        )
    
    def _flatten_prov(self, prov):
        """Flatten nested prov lists into a single list of PaperProv objects."""
        if not prov:
            return None
        
        flattened = []
        
        def flatten_recursive(item):
            if isinstance(item, list):
                for sub_item in item:
                    flatten_recursive(sub_item)
            elif item is not None:
                flattened.append(item)
        
        flatten_recursive(prov)
        return flattened if flattened else None
    
    def _chunk_by_table(self, tables: List) -> List[PaperChunk]:
        """Chunk tables into individual chunks.
        
        Each table becomes a single chunk with its caption prepended.
        No size restrictions - tables are kept intact regardless of size.
        
        Args:
            tables: List of PaperTable objects
            
        Returns:
            List of PaperChunk objects, one per table
        """
        chunks = []
        
        if not tables:
            return chunks
        
        for idx, table in enumerate(tables):
            # Combine caption with table content
            caption_text = ""
            if table.caption:
                # Caption might be a list of strings, join them
                if isinstance(table.caption, list):
                    caption_text = " ".join(table.caption)
                else:
                    caption_text = str(table.caption)
            
            # Format the chunk text
            if caption_text:
                full_text = f"{caption_text}\n{table.content}"
            else:
                full_text = table.content
            
            # Create chunk for this table
            word_count = len(full_text.split())
            
            chunk = PaperChunk(
                text=full_text,
                metadata=ChunkMetadata(
                    chunk_id=f"table_{idx+1}",
                    start_char=0,
                    end_char=len(full_text),
                    word_count=word_count,
                    section_heading=table.label if table.label else f"Table {idx+1}",
                    prov=table.prov
                ),
                arxiv_id=""  # Set by caller
            )
            
            chunks.append(chunk)
            logger.debug(f"Created chunk for table {idx+1}: {word_count} words")
        
        return chunks