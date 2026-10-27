"""Importable preprocessing utilities used by serialized Khetha pipelines."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


LOGGER = logging.getLogger("khetha.production")


def dense_for_median_imputation(encoded: Any) -> Any:
    if not hasattr(encoded, "toarray"):
        return encoded
    estimated_bytes = (
        int(encoded.shape[0])
        * int(encoded.shape[1])
        * np.dtype(encoded.dtype).itemsize
    )
    LOGGER.info(
        "Densifying sparse encoded data for median imputation (estimated %.2f MiB)",
        estimated_bytes / (1024 ** 2),
    )
    return encoded.toarray()
