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
            yield {'time': row.iloc[0], 'sensor': row.iloc[1], 'index': row.iloc[2]}

class LazyData:
    def __init__(self, filepath, data, chunk_size=100):
        self.data = []
        self.filepath = filepath
        self.chunk_size = chunk_size
        self._len = 0
        if data == "laser":
            self.generator = parse_laser_data(filepath, chunk_size)
        elif data == "drs":
            self.generator = parse_drs_data(filepath, chunk_size)
        else:
            raise ValueError(f"Invalid data type: {data}")
        self.__len__()

    def __getitem__(self, idx):
        while len(self.data) <= idx:
            try:
                self.data.append(next(self.generator))
            except StopIteration:
                raise IndexError("Index out of range in laser data")
        return self.data[idx]

    def __len__(self):
        if self._len == 0:
            chunk = 1024 * 1024
            f = np.memmap(self.filepath)
            num_newlines = sum(np.sum(f[i:i + chunk] == ord('\n'))
                               for i in range(0, len(f), chunk))
            del f
            self._len = num_newlines
        return self._len
