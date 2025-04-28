import pandas as pd
import numpy as np


def parse_laser_data(filepath, chunk_size):
    for chunk in pd.read_csv(filepath, header=None, sep="\t", chunksize=chunk_size):
        for _, row in chunk.iterrows():
            yield {
                'time_laser': row.iloc[0],
                'laser_values': row.iloc[1:].to_numpy()
            }

def parse_drs_data(filepath, chunk_size):
    for chunk in pd.read_csv(filepath, header=None, sep="\t", chunksize=chunk_size):
        for _, row in chunk.iterrows():
            yield {'time_vs': row.iloc[0], 'velocity': row.iloc[1], 'steering': row.iloc[2]}

def parse_gps_data(filepath, chunk_size):
    for chunk in pd.read_csv(filepath, header=None, sep=",", chunksize=chunk_size):
        for _, row in chunk.iterrows():
            yield {'time_vs': row.iloc[0], 'latitude': row.iloc[1], 'longitude': row.iloc[2]}

def parse_manager_data(filepath, chunk_size):
    for chunk in pd.read_csv(filepath, header=None, sep="\t", chunksize=chunk_size):
        for _, row in chunk.iterrows():
            yield {'time': row.iloc[0], 'sensor': row.iloc[1], 'index': int(row.iloc[2])}


class LazyData:
    def __init__(self, filepath, data, chunk_size=100):
        self.data = []
        self.filepath = filepath
        self.chunk_size = chunk_size
        self._len = None

        # buffer of the last chunk_size rows
        self._buf = []
        # global index of the first row in self._buf (1-based)
        self._start = 1
        if data == "laser":
            self._generator = parse_laser_data(filepath, chunk_size)
        elif data == "drs":
            self._generator = parse_drs_data(filepath, chunk_size)
        elif data == 'gps':
            self._generator = parse_gps_data(filepath, chunk_size)
        elif data == 'manager':
            self._generator = parse_manager_data(filepath, chunk_size)
        else:
            raise ValueError(f"Invalid data type: {data}")

    def __getitem__(self, idx):
        # 1-based indexing
        if idx < 1 or idx > len(self):
            raise IndexError(f"Index {idx} out of range [1..{len(self)}]")

        # if they ask for something we've already dropped → error
        if idx < self._start:
            raise IndexError(f"Index {idx} was dropped from buffer")

        # keep pulling rows until the buffer covers idx
        while self._start + len(self._buf) <= idx:
            try:
                row = next(self._generator)
            except StopIteration:
                raise IndexError("Reached end of file unexpectedly")
            self._buf.append(row)

            # once we exceed our window, pop the oldest
            if len(self._buf) > self.chunk_size:
                self._buf.pop(0)
                self._start += 1

        # now idx is inside [self._start, self._start+len(_buf)-1]
        buf_idx = idx - self._start
        return self._buf[buf_idx]

    def __len__(self):
        if self._len is None:
            chunk = 1024 * 1024
            f = np.memmap(self.filepath)
            num_newlines = sum(np.sum(f[i:i + chunk] == ord('\n'))
                               for i in range(0, len(f), chunk))
            del f
            self._len = num_newlines
        return self._len

    def __iter__(self):
        # start iteration at first element
        self._iter_idx = 1
        return self

    def __next__(self):
        # fetch next element or stop
        if self._iter_idx > len(self):
            raise StopIteration
        item = self[self._iter_idx]
        self._iter_idx += 1
        return item