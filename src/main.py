import asyncio
from pathlib import Path
from services.parser_service.parser import PDFParserService
from schemas.parser.models import PdfContent

from services.arxiv_service.instance import make_arxiv_client
from config import get_settings

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

if __name__ == "__main__":
    # test arxiv service
    arxiv_client = make_arxiv_client()
    papers = asyncio.run(arxiv_client.fetch_papers())
    print(f"Fetched {len(papers)} papers from arXiv.")

    # save fetched pdf to data_test folder
    for paper in papers:
        print(f"Downloading PDF for paper: {paper.title}")
        pdf_path = asyncio.run(arxiv_client.download_pdf(paper))
        print(f"PDF saved to: {pdf_path}")
    






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
