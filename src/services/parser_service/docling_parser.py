import logging
from pathlib import Path
from typing import List, Optional

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from schemas.parser.models import ParserType, PaperSection, PaperFigure, PaperTable, PdfContent

logger = logging.getLogger(__name__)

class DoclingParser:
    def __init__(self, max_pages: int, max_file_size_mb: int, ocr_option: bool = False, table_extraction: bool = True):
        """ Initialize the DoclingParser with specified options.

        Args:
            max_pages (int): Maximum number of pages to process.
            max_file_size_mb (int): Maximum file size in MB.
            ocr_option (bool): Enable OCR processing mode.
            table_extraction (bool): Enable table extraction mode.
        """
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.ocr_option = ocr_option
        self.table_extraction = table_extraction

        pipeline_options = PdfPipelineOptions(
            # ocr_options=EasyOcrOptions(enabled=self.ocr_option),
            do_table_structure=table_extraction,
            do_ocr=ocr_option
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self._warm_up = False

    def _warm_up_model(self):
        """ Pre-warm model with a smaill dummy document to avoid cold start -> reduce latency on first real request."""
        if not self._warm_up:
            # Only once per DoclingParser instance
            self._warm_up = True

    def _validate_pdf(self, file_path: Path) -> bool:
        """ Validate if the PDF file is not corrupted and can be opened, satisfying file_size and max_pages.

        Args:
            file_path (Path): Path to the PDF file.
        Returns:
            bool: True if the PDF is valid, False otherwise. 
        """

        try:
            # Check file exists and not empty
            if not file_path.exists() or file_path.stat().st_size == 0:
                logger.error(f"File {file_path} does not exist or is empty.")
                return False
            
            # Check file size limit
            file_size = file_path.stat().st_size  # In Bytes unit
            if file_size > self.max_file_size_bytes:
                logger.warning(f"File {file_path} exceeds the maximum file size of {self.max_file_size_bytes / 1024 / 1024} MB, skip processing.")
                return False
            
            # Check page limit
            pdf_reader = pdfium.PdfDocument(str(file_path))
            num_pages = len(pdf_reader)
            pdf_reader.close()
            if num_pages > self.max_pages:
                logger.warning(f"File {file_path} has {num_pages} pages, exceeding the maximum of {self.max_pages} pages, skip processing.")
                return False
            
            # Check if valid file
            with open(file_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    logger.error(f"File {file_path} is not a valid with PDF header.")
                    return False
                
            return True

        except Exception as e:
            logger.error(f"Error validating PDF file {file_path}: {e}")
            return False

    async def _parse(self, file_path: Path) -> Optional[str]:
        """ Parse the PDF file and extract text content.

        Args:
            file_path (Path): Path to the PDF file.
        Returns:
            Optional[str]: Extracted text content or None if parsing fails.
        """
        # Validate PDF first
        self._validate_pdf(file_path)

        # Warm up model on first use
        self._warm_up_model()


        # parse the document
        try:
            result = self._converter.convert(str(file_path), max_num_pages=self.max_pages, max_file_size=self.max_file_size_bytes)

            # extract structure
            document = result.document

            # extract sections from document structure of docling
            sections = []
            current_section = {"title": "Content", "content": ""}

            for element in document.texts:
                if hasattr(element, "label") and element.label in ["title", "section_header"]:
                    # save previous section if exists
                    if current_section["content"].strip():
                        sections.append(current_section)
                    # start new section
                    current_section = {"title": element.text.strip(), "content": ""}
                else:
                    # add content to current section
                    if hasattr(element, "text") and element.text:
                        current_section["content"] += element.text + "\n"

            # save last section
            if current_section["content"].strip():
                sections.append(PaperSection(title=current_section["title"], content=current_section["content"].strip()))

            return PdfContent(
                sections = sections,
                figures = [],
                tables = [],
                raw_text = document.export_to_text(),
                references = [],
                parser_type = ParserType.DOCLING,
                metadata = {"source": "docling", "note": "content extracted from pdf arXiv API", "file": str(file_path.name)}
            ) 

        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {e}")
            return None
