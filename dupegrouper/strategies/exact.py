"""Perform exact deduplication"""

import logging
from typing_extensions import override

from dupegrouper.definitions import DataFrameType
from dupegrouper.frames import DataFrameContainer
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# EXACT:


class Exact(DeduplicationStrategy):

    @override
    def dedupe(self, attr: str, /) -> DataFrameType:
        logger.debug(f'Deduping attribute "{attr}" with {self.__class__.__name__}()')
        frame_methods: DataFrameContainer = self.assign_group_id(attr)
        return frame_methods.frame
