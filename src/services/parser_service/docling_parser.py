import logging
from pathlib import Path
from typing import List, Optional

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from schemas.parser.models import ParserType, PaperSection, PaperFigure, PaperTable, PdfContent, PaperProv, PaperBbox, PaperPage, PaperPageSize
from exceptions import PDFValidationException, PDFParsingException

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

    def _extract_section(self, document: any) -> List[PaperSection]:
        """ Extract sections from the document structure of Docling.

        Args:
            document (any): Document object from Docling conversion result.
        Returns:
            List[PaperSection]: List of extracted sections.
        """
        sections: List[PaperSection] = []
        current_section: Optional[PaperSection] = None

        for element in document.texts:
            element_label = getattr(element, 'label', None)
            if element_label:
                # save the previous section if exists
                if current_section and current_section.content.strip():
                    sections.append(current_section)
                
                # get provenance information
                provs = []
                if (hasattr(element, "prov")) and element.prov:
                    for p in element.prov:
                        provs.append(
                            PaperProv(
                                page_no=p.page_no,
                                bbox=PaperBbox(**p.bbox.model_dump()),
                                charspan=p.charspan if hasattr(p, "charspan") else None
                            )
                        )

                # start a new section
                current_section = PaperSection(
                    label=element.label,
                    content=element.text.strip() if getattr(element, 'text', None) else "",
                    prov=provs if provs else None,
                    level=getattr(element, 'level', None) # default to None if not exists
                )
            # no label, just append content to current section
            else:
                if hasattr(element, 'text') and element.text:
                    if not current_section:
                        current_section = PaperSection(
                            label="",
                            content="",
                            prov=None,
                            level=None
                        )
                    current_section.content += element.text + "\n"

        # append the last section if exists
        if current_section and current_section.content.strip():
            sections.append(current_section)

        return sections

    def _extract_tables(self, document: any, sections: List[PaperSection]) -> List[PaperTable]:
        """ Extract tables from the document structure of Docling.

        Args:
            document (any): Document object from Docling conversion result.
        Returns:
            List[PaperTable]: List of extracted tables.
        """
        import pandas as pd

        tables: List[PaperTable] = []

        for table in document.tables:
            caption = getattr(table, 'captions', None)
            # get the caption of table from referenced section (raw data it is list of dict)
            if caption:
                captions = []
                for cap in caption:
                    _cref = cap.cref
                    if _cref.startswith('#/texts'):
                        text_id = int(_cref.split('/')[-1])  # get the text id referenced index
                        caption_text = sections[text_id].content if text_id < len(sections) else ""
                        captions.append(caption_text)
                        break

            provs = []
            if (hasattr(table, "prov")) and table.prov:
                for p in table.prov:
                    provs.append(
                        PaperProv(
                            page_no=p.page_no,
                            bbox=PaperBbox(**p.bbox.model_dump()),
                            charspan=p.charspan if hasattr(p, "charspan") else None
                        )
                    )

            table_df: pd.DataFrame = table.export_to_dataframe(doc=document)

            tables.append(
                PaperTable(
                    label=getattr(table, 'label', 'TABLE'),
                    prov=provs,
                    caption=captions if captions else None,
                    content=table_df.to_markdown(index=False)  # Convert DataFrame to markdown string  
                )
            )
                 
        return tables

    def _extract_pages(self, document: any) -> List[PaperPage]:
        """ Extract pages from the document structure of Docling.

        Args:
            document (any): Document object from Docling conversion result.
        Returns:
            List[PaperPage]: List of extracted pages.
        """

        pages: List[PaperPage] = [
            PaperPage(
                page_no=p.page_no,
                page_size=PaperPageSize(width=p.size.width, height=p.size.height),
                image=p.image if hasattr(p, "image") else None
            )
            for p in document.pages.values()
        ]

        return pages
    
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

            # ------------------------- extract content include heading and paragraphs -------------------------
            sections = self._extract_section(document)

            # --------------------------------------- extract tables -------------------------------------------
            tables = self._extract_tables(document, sections) if self.table_extraction else []

            # --------------------------------- extract pages information --------------------------------------
            pages = self._extract_pages(document)

            # drop sections with label = caption, page_header, foot_note
            sections = [sec for sec in sections if sec.label.lower() not in ["caption", "page_header", "foot_note"]]

            return PdfContent(
                sections=sections,
                tables=tables,
                figures=[],  # Unable to extract figures for now
                raw_text=document.export_to_text(),
                page_info=pages,
                parser_type=ParserType.DOCLING,
                metadata={
                    "source": "docling parser",
                    "note": "Content extracted from PDF, metadata comes from arXiv API"
                }
            )

        except PDFValidationException as e:
            error_msg = str(e).lower()
            if "too large" in error_msg:
                logger.error(f"PDF file {file_path} is too large to process: {e}. Skip processing.")
                return None
            elif "too many pages" in error_msg:
                logger.error(f"PDF file {file_path} has too many pages to process: {e}. Skip processing.")
                return None
            else:
                raise
            
        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {e}")
            error_msg = str(e).lower()

            if "not valid" in error_msg:
                logger.error("PDF appears to be corrupted or not a valid PDF file")
                raise PDFParsingException(f"PDF appears to be corrupted or invalid: {file_path}")
            elif "timeout" in error_msg:
                logger.error("PDF processing timed out - file may be too complex")
                raise PDFParsingException(f"PDF processing timed out: {file_path}")
            elif "memory" in error_msg or "ram" in error_msg:
                logger.error("Out of memory - PDF may be too large or complex")
                raise PDFParsingException(f"Out of memory processing PDF: {file_path}")
            elif "max_num_pages" in error_msg or "page" in error_msg:
                logger.error(f"PDF processing issue likely related to page limits (current limit: {self.max_pages} pages)")
                raise PDFParsingException(
                    f"PDF processing failed, possibly due to page limit ({self.max_pages} pages). Error: {e}"
                )
            else:
                raise PDFParsingException(f"Failed to parse PDF with Docling: {e}")

