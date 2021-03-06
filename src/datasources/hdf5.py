"""Copyright (c) 2020 AIT Lab, ETH Zurich

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

"""HDF5 data source for gaze estimation."""
from threading import Lock
from typing import List

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf

from core import BaseDataSource
import logging
logger = logging.getLogger(__name__)

class HDF5Source(BaseDataSource):
    """HDF5 data loading class (using h5py)."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 hdf_path: str,
                 use_colour: bool=False,
                 keys_to_use: List[str]=None,
                 entries_to_use: List[str]=None,
                 testing=False,
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys."""
        hdf5 = h5py.File(hdf_path, 'r')
        self._short_name = 'HDF:%s' % '/'.join(hdf_path.split('/')[-2:])
        if testing:
            self._short_name += ':test'

        # Cache other settings
        self._use_colour = use_colour

        # Create global index over all specified keys
        if keys_to_use is None:  # use all available keys if not specified
            keys_to_use = list(hdf5.keys())
        self._index_to_key = {}
        index_counter = 0
        for key in keys_to_use:
            n = next(iter(hdf5[key].values())).shape[0]
            for i in range(n):
                self._index_to_key[index_counter] = (key, i)
                index_counter += 1
        self._num_entries = index_counter

        if entries_to_use is None:  # use all available input data if not specified
            entries_to_use = list(next(iter(hdf5.values())).keys())
        self.entries_to_use = entries_to_use

        self._hdf5 = hdf5
        self._mutex = Lock()
        self._current_index = 0
        super().__init__(tensorflow_session, batch_size=batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0

    @property
    def num_entries(self):
        """Number of entries in this data source."""
        return self._num_entries

    @property
    def short_name(self):
        """Short name specifying source HDF5."""
        return self._short_name

    def cleanup(self):
        """Close HDF5 file before running base class cleanup routine."""
        super().cleanup()

    def reset(self):
        """Reset index."""
        with self._mutex:
            super().reset()
            self._current_index = 0

    def entry_generator(self, yield_just_one=False):
        """Read entry from HDF5."""
        try:
            while range(1) if yield_just_one else True:
                with self._mutex:
                    if self._current_index >= self.num_entries:
                        if self.testing:
                            break
                        else:
                            self._current_index = 0
                    current_index = self._current_index
                    self._current_index += 1

                key, index = self._index_to_key[current_index]
                data = self._hdf5[key]
                entry = {}
                for name in self.entries_to_use:
                    entry[name] = data[name][index, :]
                yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Normalize image intensities."""
        for k, v in entry.items():
            # if v.ndim == 3: # We get histogram-normalized BGR inputs
            if k in ['left-eye', 'right-eye', 'face', 'eye-region']:  # We get histogram-normalized BGR inputs
                # debug and check dimension
                # if self.testing:
                #     logger.info("Before expanding, shape {}. k name {}".format(v.shape, k))
                # for currupted test data
                # if self.testing:
                #     logger.info("Before converting, shape {}. k name {}".format(v.shape, k))
                    # v = np.expand_dims(v, axis = -1)
                    # v = np.concatenate((v,v,v), axis = 2)
                    # # if self.testing:
                    # logger.info("After converting shape {}. k name {}".format(v.shape, k))

                # change the currupted data into correct form --> Channel last
                if v.shape[0] == 1:
                    v = np.transpose(v, [1, 2, 0])
                    v = np.concatenate((v,v,v), axis = 2)
                    # v = cv.cvtColor(v, cv.COLOR_GRAY2RGB)

                if not self._use_colour:
                    v = cv.cvtColor(v, cv.COLOR_BGR2GRAY)
                    
                v = v.astype(np.float32)

                # v *= 2.0 / 255.0
                # v -= 1.0
                if self._use_colour and self.data_format == 'NCHW':
                    v = np.transpose(v, [2, 0, 1])
                elif not self._use_colour:
                    v = np.expand_dims(v, axis=0 if self.data_format == 'NCHW' else -1)
                entry[k] = v

        # Ensure all values in an entry are 4-byte floating point numbers
        for key, value in entry.items():
            entry[key] = value.astype(np.float32)

        return entry
