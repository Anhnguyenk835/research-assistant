import logging
from typing import List, Optional
from pathlib import Path

from schemas.parser.models import ParsedPaper, PdfContent, ParserType
from services.parser_service.docling_parser import DoclingParser

logger = logging.getLogger(__name__)

class PDFParserService:
    """ Service to parse PDF files using different parsers."""

    def __init__(self, max_pages: int, max_file_size_mb: int, ocr_option: bool = False, table_extraction: bool = True):
        """ Initialize the PDFParserService with specified options.

        Args:
            max_pages (int): Maximum number of pages to process.
            max_file_size_mb (int): Maximum file size in MB.
            ocr_option (bool): Enable OCR processing mode.
            table_extraction (bool): Enable table extraction mode.
        """
        self.docling_parser = DoclingParser(max_pages, max_file_size_mb, ocr_option, table_extraction)

    async def parse_pdf(self, file_path: Path) -> Optional[PdfContent]:
        """ Parse the PDF file and extract its content.

        Args:
            file_path (Path): Path to the PDF file.
        Returns:
            Optional[PdfContent]: Extracted content from the PDF or None if parsing failed.
        """

        if not file_path.exists():
            logger.error(f"File {file_path} does not exist.")
            return None
        
        try:
            parsed_content = await self.docling_parser._parse(file_path)
            if parsed_content:
                logger.info(f"Successfully parsed PDF {file_path}")
                return parsed_content
            else:
                logger.error(f"Docling parsing returned no result for {file_path.name}")
                return None
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            return None
