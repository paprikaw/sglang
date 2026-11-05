from enum import IntEnum, unique


@unique
class P2PTag(IntEnum):
    """
    Tags reserved for point-to-point communication protocols.

    Communications introduced outside existing scheduler loops need explicit
    tags to avoid being consumed by unrelated send/recv paths.
    """

    DEFAULT = 0
    HIRADIX_PP_SYNC = int.from_bytes(b"PpHi", byteorder="big")
    HIRADIX_PP_PREFETCH_ADMISSION = int.from_bytes(b"PpPa", byteorder="big")
    HIRADIX_PP_PREFIX_ALIGNMENT = int.from_bytes(b"PpMa", byteorder="big")
    GRAMMAR_PP_SYNC = int.from_bytes(b"PpGr", byteorder="big")
