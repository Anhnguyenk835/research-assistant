import asyncio
from pathlib import Path
import pandas as pd
from services.parser_service.parser import PDFParserService
from schemas.parser.parser_models import PdfContent
from schemas.arxiv.arxiv_models import ArxivPaper

from services.arxiv_service.instance import make_arxiv_client
from services.parser_service.instance import get_pdf_parser_service
from services.parser_service.parser import PDFParserService
from services.indexing.chunker import HeadingChunk
from config import get_settings

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from ollama import chat, ChatResponse

if __name__ == "__main__":
    # test arxiv service
    # arxiv_client = make_arxiv_client()
    # papers = asyncio.run(arxiv_client.fetch_papers())

    # # papers = List[ArxivPaper]
    # # save to csv
    

    # df = pd.DataFrame([paper.dict() for paper in papers])
    # df.to_csv("arxiv_papers.csv", index=False)

    # print(f"Fetched {len(papers)} papers from arXiv.")

    # # save fetched pdf to data_test folder
    # for paper in papers:
    #     print(f"Downloading PDF for paper: {paper.title}")
    #     pdf_path = asyncio.run(arxiv_client.download_pdf(paper))
    #     print(f"PDF saved to: {pdf_path}")

    # test pdf parser service

    arxiv_data = ArxivPaper(
        arxiv_id="2510.03230v1",
        title="Sample Paper Title",
        authors=["Author 1", "Author 2"],
        abstract="This is a sample abstract.",
        categories=["cs.AI"],
        published_date="2024-10-04",
        pdf_url="data_test/2402.05131v2.pdf"
    )

    parser = get_pdf_parser_service()
    chunker = HeadingChunk(max_chunk_size=400, min_chunk_size=100)
    pdf_path = Path("data_test/2402.05131v2.pdf")
    parsed_paper = asyncio.run(parser.parse_pdf(arxiv_data))
    print(f"Parsed paper successfully.")
    # SAVE THE PARSE PAPER SECTION AND TABLE TO CSV
    df_sections = pd.DataFrame([section.dict() for section in parsed_paper.content.sections])
    df_sections.to_csv("parsed_sections_2.csv", index=False)
    print(f"Saved parsed sections to parsed_sections_2.csv")
    df_tables = pd.DataFrame([table.dict() for table in parsed_paper.content.tables])
    df_tables.to_csv("parsed_tables_2.csv", index=False)
    print(f"Saved parsed tables to parsed_tables_2.csv")

    print(f"chunking paper...")
    chunks = chunker.chunk_paper(parsed_paper)
    print(f"Created {len(chunks)} chunks.")

    # save chunks to csv
    df = pd.DataFrame([{
        "chunk_id": chunk.metadata.chunk_id,
        "text": chunk.text,
        "start_char": chunk.metadata.start_char,
        "end_char": chunk.metadata.end_char,
        "word_count": chunk.metadata.word_count,
        "section_heading": chunk.metadata.section_heading,
        "prov": chunk.metadata.prov,
        "arxiv_id": chunk.arxiv_id
    } for chunk in chunks])
    df.to_csv("paper_chunks_2.csv", index=False)

    # # save to csv
    # df = pd.DataFrame([section.dict() for section in parsed_paper.sections])
    # df.to_csv("parsed_paper.csv", index=False)

