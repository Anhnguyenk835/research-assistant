import logging
from typing import List, Optional
from pathlib import Path

from schemas.parser.parser_models import ParsedPaper, PdfContent, ParserType
from schemas.arxiv.arxiv_models import ArxivPaper
from services.parser_service.docling_parser import DoclingParser
from exceptions import PDFParsingException, PDFValidationException

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

    async def parse_pdf(self, arxiv_data: ArxivPaper) -> Optional[ParsedPaper]:
        """ Parse the PDF file and extract its content.

        Args:
            file_path (Path): Path to the PDF file.
        Returns:
            Optional[PdfContent]: Extracted content from the PDF or None if parsing failed.
        """

        if not Path(arxiv_data.pdf_url).exists():
            logger.error(f"File {arxiv_data.pdf_url} does not exist.")
            return None
        
        try:
            parsed_content = await self.docling_parser._parse(arxiv_data)
            if parsed_content:
                logger.info(f"Successfully parsed PDF {arxiv_data.pdf_url}")
                return parsed_content
            else:
                logger.error(f"Docling parsing returned no result for {arxiv_data.pdf_url.name}")
                raise PDFParsingException(f"Docling parsing returned no result for {arxiv_data.pdf_url.name}")

        except (PDFValidationException, PDFParsingException):
            raise
            
        except Exception as e:
            logger.error(f"Error parsing PDF {arxiv_data.pdf_url}: {e}")
            raise PDFParsingException(f"Error parsing PDF with docling {arxiv_data.pdf_url}: {e}")
