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

"""Object to log all data that it is given.
"""

import timeit
from collections import deque
from typing import List, Dict, Union, Any
import csv
import os

import torch

CsvRowtype = Dict[str, Union[int, float]]


class Logger():

    def __init__(self,
                 modes: List[str],
                 sources: List[str],
                 directory: str,
                 csv_chunksize: int = 1000,
                 tb_chunksize: int = 1000):

        self.function_map = {'csv': self._write_csv_buffered,
                             'tb': self._write_tb}

        assert all(m in self.function_map.keys() for m in modes)

        self._modes = modes
        self._sources = sources
        self._directory = directory
        self._csv_chunksize = csv_chunksize
        self._tb_chunksize = tb_chunksize

        self._filepaths = self._gen_filepaths()
        self._buffers = self._gen_buffers()
        self._csv_headers = {}

        for m, fp in self._filepaths.items():
            print("%s logs will be saved at %s" % (m, fp))

    def _gen_buffers(self) -> Dict[str, Dict[str, List[CsvRowtype]]]:
        return {m: {s: [] for s in self._sources} for m in self._modes}

    def _gen_filepaths(self) -> Dict[str, str]:
        filepaths = {m: "/".join([self._directory, m]) for m in self._modes}
        for fp in filepaths.values():
            os.makedirs(fp, exist_ok=True)
        return filepaths

    def log(self,
            source: str,
            log_data: Dict[str, Any]):

        prepped_data = self._prep_data(log_data)

        for m in self._modes:
            self.function_map[m](source, prepped_data)

    def _prep_data(self,
                   log_data: Dict[str, Any]) -> CsvRowtype:
        if 'frame' in log_data.keys():
            del log_data['frame']

        for k, v in log_data.items():
            if isinstance(v, torch.Tensor):
                log_data[k] = v.detach().cpu().numpy().flatten()[0]

        return log_data

    def _write_tb(self, path: str, source: str, log_data: Dict[str, Any]):
        pass

    def write_buffers(self):
        for s in self._sources:
            self._write_buffer(s)

    def _write_buffer(self, source):
        path = self._filepaths['csv']
        filename = "/".join([path, source]) + ".csv"

        buffer = self._buffers['csv'][source]
        rows = buffer[:self._csv_chunksize]
        del buffer[:self._csv_chunksize]

        if rows:
            self._write_csv_rows(filename, rows)

    def _write_csv_buffered(self,
                            source: str,
                            log_data: Dict[str, Any]):
        buffer = self._buffers['csv'][source]
        buffer.append(log_data)

        if len(buffer) >= self._csv_chunksize:
            self._write_buffer(source)

    def _get_header(self,
                    filename: str,
                    csv_columns: List[str]) -> List[str]:

        # If csv file does not exist, create it
        if not os.path.isfile(filename):
            csv_header = csv_columns
            self._write_csv_header(filename, csv_columns)
        # Else, open it
        else:
            with open(filename) as f:
                reader = csv.reader(f, delimiter=';')
                csv_header = next(reader)

        return csv_header

    def _write_csv_rows(self,
                        filename: str,
                        rows: List[CsvRowtype]):
        """Wrapper for writing rows of data into a csv file
        """
        # get the csv header csv columns
        csv_columns = rows[0].keys()

        if filename not in self._csv_headers.keys():
            self._csv_headers[filename] = self._get_header(
                filename, csv_columns)

        try:
            with open(filename, 'a') as csvfile:
                writer = csv.DictWriter(csvfile,
                                        delimiter=';',
                                        fieldnames=self._csv_headers[filename])
                writer.writerows(rows)
        except IOError:
            print("I/O error")

    def _write_csv_header(self,
                          filename: str,
                          header: List[str]):
        """Wrapper for writing only the header into a csv file
        """
        try:
            with open(filename, 'w') as csvfile:
                writer = csv.DictWriter(csvfile,
                                        delimiter=';',
                                        fieldnames=header)
                writer.writeheader()
        except IOError:
            print("I/O error")


""" test_row = {
    "A": 1, "B": 2
}
logger = Logger(['csv'], ['episodes'], './logs', csv_chunksize=1)
print(timeit.timeit(lambda: logger.log('training', test_row), number=1))


logger = Logger(['csv'], ['episodes'], './logs', csv_chunksize=1000)
print(timeit.timeit(lambda: logger.log('training', test_row), number=1000000))
 """
