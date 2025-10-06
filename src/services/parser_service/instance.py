from functools import lru_cache
from .parser import PDFParserService

@lru_cache(maxsize=1)
def get_pdf_parser_service() -> PDFParserService:
    """ Factory function to get a cached instance of PDFParserService. """
    return PDFParserService(
        max_pages=25, 
        max_file_size_mb=5, 
        ocr_option=False, 
        table_extraction=True
    )
