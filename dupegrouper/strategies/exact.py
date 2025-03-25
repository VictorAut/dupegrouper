import logging
from typing_extensions import override

from dupegrouper.definitions import frames
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# EXACT:


class Exact(DeduplicationStrategy):

    @override
    def dedupe(self, attr: str, /) -> frames:
        logger.debug(f'Deduping attribute "{attr}" with {self.__class__.__name__}()')
        return self._assign_group_id(attr).frame
