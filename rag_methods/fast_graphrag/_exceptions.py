class InvalidStorageError(Exception):
    """Exception raised for errors in the storage operations."""

    def __init__(self, message: str = "Invalid storage operation"):
        self.message = message
        super().__init__(self.message)


class InvalidStorageUsageError(Exception):
    """Exception raised for errors in the usage of the storage."""

    def __init__(self, message: str = "Invalid usage of the storage"):
        self.message = message
        super().__init__(self.message)


class LLMServiceNoResponseError(Exception):
    """Exception raised when the LLM service does not provide a response."""

    def __init__(self, message: str = "LLM service did not provide a response"):
        self.message = message
        super().__init__(self.message)
