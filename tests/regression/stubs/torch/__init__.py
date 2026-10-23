"""Test-only stand-in for torch (no GPU). See ../README.md."""


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def empty_cache():
        pass

    @staticmethod
    def get_device_name(i=0):
        return "none"

    @staticmethod
    def get_device_properties(i=0):
        class P:
            total_memory = 0

        return P()


cuda = _Cuda()


class Tensor:  # referenced by scipy/sklearn array-API type checks
    pass


class dtype:
    pass


class device:
    def __init__(self, *a, **k):
        pass
