import asyncio
from pathlib import Path
import pandas as pd
from schemas.parser.parser_models import ParsedPaper
from schemas.arxiv.arxiv_models import ArxivPaper
from schemas.indexing.indexing_models import PaperChunk

from services.arxiv_service.instance import make_arxiv_client
from services.parser_service.instance import get_pdf_parser_service
from services.indexing_service.instance import make_hybrid_indexing_service
from config import get_settings

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from typing import List

from ollama import chat, ChatResponse

if __name__ == "__main__":
    # 1. test arxiv service
    arxiv_client = make_arxiv_client()
    papers: List[ArxivPaper] = asyncio.run(arxiv_client.fetch_papers())
    # download pdfs and replace pdf_url with local path
    for paper in papers:
        print(f"Downloading PDF for paper: {paper.title}")
        pdf_path = asyncio.run(arxiv_client.download_pdf(paper))
        print(f"PDF saved to: {pdf_path}")
        paper.pdf_url = str(pdf_path)
    # save papers to csv
    df = pd.DataFrame([paper.model_dump() for paper in papers])
    df.to_csv("arxiv_papers.csv", index=False)

    # 2. test pdf parser service
    parser = get_pdf_parser_service()   
    for paper in papers:
        parsed_paper: ParsedPaper = asyncio.run(parser.parse_pdf(paper))
    # save the parsed paper
    df_parsed_metadata = pd.DataFrame([parsed_paper.metadata.model_dump()])
    df_parsed_metadata.to_csv("parsed_paper_metadata.csv", index=False)
    df_parsed_sections = pd.DataFrame([section.model_dump() for section in parsed_paper.content.sections])
    df_parsed_sections.to_csv("parsed_paper_sections.csv", index=False)
    df_parsed_tables = pd.DataFrame([table.model_dump() for table in parsed_paper.content.tables])
    df_parsed_tables.to_csv("parsed_paper_tables.csv", index=False)

    # 3. test indexing service
    settings = get_settings()
    indexing_service = make_hybrid_indexing_service(settings=settings)

    # 3.1. chunk the paper
    # chunks: List[PaperChunk] = asyncio.run(indexing_service.chunker.chunk_paper(parsed_paper))
    # # save chunks to csv
    # df_chunks = pd.DataFrame([{
    #     "chunk_id": chunk.metadata.chunk_id,
    #     "start_char": chunk.metadata.start_char,
    #     "end_char": chunk.metadata.end_char,
    #     "word_count": chunk.metadata.word_count,
    #     "section_heading": chunk.metadata.section_heading,
    #     "prov": chunk.metadata.prov,
    #     "text": chunk.text,
    #     "arxiv_metadata": chunk.arxiv_metadata.model_dump() if chunk.arxiv_metadata else None
    # } for chunk in chunks])
    # df_chunks.to_csv("paper_chunks.csv", index=False)

    # 3.2. embed the chunks
    # 3.3. index the chunks
    result = asyncio.run(indexing_service.index_parsed_paper(parsed_paper))
    print(f"Indexing result: {result}")


    #-----------------------------------------------------------------------------------------------

    # 3. test chunking
    # chunker = HeadingChunk(max_chunk_size=400, min_chunk_size=100)
    # chunks: List[PaperChunk] = chunker.chunk_paper(parsed_paper)
    # # save chunks to csv
    # df_chunks = pd.DataFrame([{
    #     "chunk_id": chunk.metadata.chunk_id,
    #     "start_char": chunk.metadata.start_char,
    #     "end_char": chunk.metadata.end_char,
    #     "word_count": chunk.metadata.word_count,
    #     "section_heading": chunk.metadata.section_heading,
    #     "prov": chunk.metadata.prov,
    #     "text": chunk.text,
    #     "arxiv_metadata": chunk.arxiv_metadata.model_dump() if chunk.arxiv_metadata else None
    # } for chunk in chunks])
    # df_chunks.to_csv("paper_chunks.csv", index=False)  








    



    # test pdf parser service
    # arxiv_data = ArxivPaper(
    #     arxiv_id="2510.03230v1",
    #     title="Sample Paper Title",
    #     authors=["Author 1", "Author 2"],
    #     abstract="This is a sample abstract.",
    #     categories=["cs.AI"],
    #     published_date="2024-10-04",
    #     pdf_url="data_test/2402.05131v2.pdf"
    # )

    # parser = get_pdf_parser_service()

    # print(f"chunking paper...")
    # chunks = chunker.chunk_paper(parsed_paper)
    # print(f"Created {len(chunks)} chunks.")

    # # save chunks to csv
    # df = pd.DataFrame([{
    #     "chunk_id": chunk.metadata.chunk_id,
    #     "text": chunk.text,
    #     "start_char": chunk.metadata.start_char,
    #     "end_char": chunk.metadata.end_char,
    #     "word_count": chunk.metadata.word_count,
    #     "section_heading": chunk.metadata.section_heading,
    #     "prov": chunk.metadata.prov,
    #     "arxiv_id": chunk.arxiv_id
    # } for chunk in chunks])
    # df.to_csv("paper_chunks_2.csv", index=False)

    # # save to csv
    # df = pd.DataFrame([section.dict() for section in parsed_paper.sections])
    # df.to_csv("parsed_paper.csv", index=False)

