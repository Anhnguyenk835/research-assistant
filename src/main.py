import asyncio
from pathlib import Path
from services.parser_service.parser import PDFParserService
from schemas.parser.models import PdfContent

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

if __name__ == "__main__":
    parser_service = PDFParserService(max_pages=25, max_file_size_mb=5, ocr_option=True, table_extraction=True)
    pdf_path = Path("../data_test/2402.05131v2.pdf")
    
    content = asyncio.run(parser_service.parse_pdf(pdf_path))
    if content:
        print("Parsed PDF Content successfully")
        # print(content.raw_text)
        # save to csv (all information of pdfContent)
        import pandas as pd
        sections = pd.DataFrame([section.dict() for section in content.sections])
        sections.to_csv("parsed_pdf_section.csv", index=False)
        tables = pd.DataFrame([table.dict() for table in content.tables])
        tables.to_csv("parsed_pdf_tables.csv", index=False)
        pdf_content = pd.DataFrame([content.dict()])
        pdf_content.to_csv("parsed_pdf_content.csv", index=False)
    else:
        print("Failed to parse PDF.")


    # pipeline_options = PdfPipelineOptions(
    #     # ocr_options=EasyOcrOptions(enabled=self.ocr_option),
    #     do_table_structure=True,
    #     # do_ocr=True
    # )

    # converter = DocumentConverter(
    #     format_options={
    #     InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # file_path = Path("../data_test/2402.05131v2.pdf")

    # result = converter.convert(str(file_path), max_num_pages=25)

    # document = result.document

    # # save to txt
    # with open("converted_document.txt", "w", encoding="utf-8") as f:
    #     f.write(str(document))
