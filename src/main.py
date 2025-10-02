import asyncio
from pathlib import Path
from services.parser_service.parser import PDFParserService
from schemas.parser.models import PdfContent

if __name__ == "__main__":
    parser_service = PDFParserService(max_pages=25, max_file_size_mb=5, ocr_option=True, table_extraction=True)
    pdf_path = Path("../data_test/2402.05131v2.pdf")
    
    content = asyncio.run(parser_service.parse_pdf(pdf_path))
    if content:
        print("Parsed PDF Content successfully")
        print(content.raw_text)
        # save to csv
        import pandas as pd
        sections = [section.dict() for section in content.sections]
        df = pd.DataFrame(sections)
        df.to_csv("parsed_sections.csv", index=False)
    else:
        print("Failed to parse PDF.")
