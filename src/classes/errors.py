class DataSetException(Exception):
    """``DataSetError`` raised by ``AbstractDataSet`` implementations
    in case of failure of input/output methods.
    ``AbstractDataSet`` implementations should provide instructive
    information in case of failure.
    """

    pass


class DataSetIOError(IOError):
    """``DataSetError`` raised by ``AbstractDataSet`` implementations
    in case of failure of input/output methods.
    ``AbstractDataSet`` implementations should provide instructive
    information in case of failure.
    """

    pass
