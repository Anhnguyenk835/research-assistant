# Parsing exceptions

class PDFValidationException(Exception):
    """ Exception raised for errors in the PDF validation process. """
    pass

class PDFParsingException(Exception):
    """ Exception raised for errors in the PDF parsing process. """
    pass