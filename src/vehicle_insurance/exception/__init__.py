import sys

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed error information including file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{line_number}] "
        f"error message: [{str(error)}]"
    )


class MyException(Exception):
    """
    Custom exception class for Vehicle Insurance MLOps project.
    """

    def __init__(self, error: Exception, error_detail: sys):
        """
        Initialize custom exception with detailed traceback info.
        """
        self.error_message = error_message_detail(error, error_detail)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message
