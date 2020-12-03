# Copyright 2020 Michael Janschek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pylint: disable=empty-docstring
"""
"""
import csv
import os
from collections import deque
from typing import Any, Deque, Dict, List, Union

import torch

CsvRowtype = Dict[str, Union[int, float]]


class Logger():
    """Object that manages the writing of logs.

    Warnings
    --------
    Tensorboard functionality not yet implemented.

    Parameters
    ----------
    sources: `list` of `str`
        A list of sources, that shall be registered. A csv file will be created for each source.
    directory: `str`
        The logs root directory.
    modes: `list` of `str`
        The modes this instance uses for logging. Possible values are ``'csv'`` and ``'tb'``.
    csv_chunksize: `int`
        The chunksize of buffered writing of csv files.
    tb_chunksize: `int`
        The chunksize of buffered writing of tensorboard files.
    """

    def __init__(self,
                 sources: List[str],
                 directory: str,
                 modes: List[str] = None,
                 csv_chunksize: int = 10,
                 tb_chunksize: int = 10):

        self.function_map = {'csv': self._write_csv_buffered,
                             'tb': self._write_tb}

        # modes must be known to logic
        assert all(m in self.function_map.keys() for m in modes)
        assert isinstance(modes, list)

        # ATTRIBUTES
        self._modes = modes
        self._sources = sources
        self._directory = directory
        self._csv_chunksize = csv_chunksize
        self._tb_chunksize = tb_chunksize

        # generate paths and storage
        self._filepaths = self._gen_filepaths()
        self._buffers = self._gen_buffers()
        self._csv_headers = {}

        for mode, filepath in self._filepaths.items():
            print("%s logs will be saved at %s" % (mode, filepath))

    def _gen_buffers(self) -> Dict[str, Dict[str, Deque[CsvRowtype]]]:
        """Return a dictionary that is intended for use as buffer for all modes known.
        """
        return {m: {s: deque() for s in self._sources} for m in self._modes}

    def _gen_filepaths(self) -> Dict[str, str]:
        """Return a dictionary containing filepaths for each mode.
        """
        filepaths = {m: "/".join([self._directory, m]) for m in self._modes}
        for filepath in filepaths.values():
            os.makedirs(filepath, exist_ok=True)
        return filepaths

    def log(self,
            source: str,
            log_data: Dict[str, Any]):
        """Prepare :py:attr:`log_data` inplace and write in all modes declared on initialization.

        Parameters
        ----------
        source: `str`
            The :py:attr:`source` this :py:attr:`log_data` shall be written to.
        log_data: `dict`
            The data that shall be logged :py:attr:`log_data`.
        """
        self._prep_data(log_data)

        for mode in self._modes:
            self.function_map[mode](source, log_data)

    def _prep_data(self,
                   log_data: Dict[str, Any]) -> CsvRowtype:
        """Clean and transform a data dictionary inplace.

        Parameters
        ----------
        log_data: `dict`
            The log data that shall be prepared.
        """
        # clean frame data for log, this is not needed
        if 'frame' in log_data.keys():
            del log_data['frame']

        # transform all tensors in log_data
        for key, value in log_data.items():
            if isinstance(value, torch.Tensor):
                log_data[key] = value.detach() \
                    .cpu() \
                    .numpy() \
                    .flatten()[0]

    def _write_tb(self,
                  source: str,
                  log_data: Dict[str, Any]):
        """Prepare :py:attr:`log_data` inplace and write to Tensorboard.

        Warnings
        --------
        Not yet implemented.

        Parameters
        ----------
        source: `str`
            The :py:attr:`source` this :py:attr:`log_data` shall be written to.
        log_data: `dict`
            The data that shall be logged :py:attr:`log_data`.
        """

    def write_buffers(self):
        """Write and clear all buffers.
        """
        for source in self._sources:
            self._write_buffer(source, True)

    def _write_buffer(self, source: str, clear: bool = False):
        """Load and write the buffer registered with :py:attr:`source`.

         Clear data that has been written.

        Parameters
        ----------
        source: `str`
            The source this data relates to. This declares the writing destination.
        clear: `bool`
            Set True if *whole* buffer shall be written and cleared
        """
        # gen filepath
        path = self._filepaths['csv']
        filename = "/".join([path, source]) + ".csv"

        # pop buffer
        buffer = self._buffers['csv'][source]
        n_rows = len(buffer) if clear else self._csv_chunksize
        rows = [buffer.popleft() for _ in range(n_rows)]

        if n_rows > 0:
            self._write_csv_rows(filename, rows)

    def _write_csv_buffered(self,
                            source: str,
                            log_data: Dict[str, Any]):
        """Write :py:attr:`log_data` to registered :py:attr:`source` with a buffer.

        Parameters
        ----------
        source: `str`
            The :py:attr:`source` this :py:attr:`log_data` shall be written to.
        log_data: `dict`
            The data that shall be logged :py:attr:`log_data`.
        """
        # append buffer
        buffer = self._buffers['csv'][source]
        buffer.append(log_data)

        # write, if buffer has enough entries
        if len(buffer) >= self._csv_chunksize:
            self._write_buffer(source)

    def _get_header(self,
                    filename: str,
                    csv_columns: List[str]) -> List[str]:
        """Write and return a valid csv header (a `list` of column names).

        If available, use header of already existing file.
        Create a header for the given list of columns and write file, otherwise.

        Parameters
        ----------
        filename: `str`
            The filename, the header would/shall be placed.
        csv_columns: `list` of `str`
            A list of column names that shall be used.
        """
        # pylint: disable=invalid-name

        # If csv file does not exist, create it
        if not os.path.isfile(filename):
            csv_header = csv_columns
            self._write_csv_header(filename, csv_columns)
        # Else, open it
        else:
            with open(filename) as f:
                reader = csv.reader(f, delimiter=',')
                csv_header = next(reader)

        return csv_header

    def _write_csv_rows(self,
                        filename: str,
                        rows: List[CsvRowtype]):
        """Wrapper for writing rows of data into a csv file.

        Invokes :py:meth:`_get_header()` to get a viable hear for wanted :py:attr:`filename`.
        Writes csv rows into this file.

        Parameters
        ----------
        filename: `str`
            The destination filename.
        rows: `list` of csv rows (`dict`)
            The data as `list`.
        """
        # get the csv header csv columns
        csv_columns = rows[0].keys()

        if filename not in self._csv_headers.keys():
            self._csv_headers[filename] = self._get_header(
                filename, csv_columns)

        try:
            with open(filename, 'a') as csvfile:
                writer = csv.DictWriter(csvfile,
                                        delimiter=',',
                                        fieldnames=self._csv_headers[filename])
                writer.writerows(rows)
        except IOError:
            print("I/O error")

    def _write_csv_header(self,
                          filename: str,
                          header: List[str]):
        """Wrapper for writing only the header into a csv file

        Parameters
        ----------
        filename: `str`
            The destination filename.
        header: `list` of `str`
            A list of column names.
        """
        try:
            with open(filename, 'w') as csvfile:
                writer = csv.DictWriter(csvfile,
                                        delimiter=',',
                                        fieldnames=header)
                writer.writeheader()
        except IOError:
            print("I/O error")
