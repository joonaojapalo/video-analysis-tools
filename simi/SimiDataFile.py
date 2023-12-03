from decimal import *
getcontext().prec = 6

DELIM = "\t"


DEFAULT_COLS = [
    "javelin: top",
    "javelin: top",
    "javelin: center",
    "javelin: center",
    "javelin: end",
    "javelin: end",
    "ref-1",
    "ref-1",
    "ref-2",
    "ref-2",
    "head",
    "head",
    "left shoulder",
    "left shoulder",
    "left elbow",
    "left elbow",
    "left hand",
    "left hand",
    "middle finger middle joint left",
    "middle finger middle joint left",
    "right shoulder",
    "right shoulder",
    "right elbow",
    "right elbow",
    "right hand",
    "right hand",
    "middle finger middle joint right",
    "middle finger middle joint right",
    "left hip",
    "left hip",
    "left knee",
    "left knee",
    "left ankle-bone",
    "left ankle-bone",
    "foot tip left",
    "foot tip left",
    "right hip",
    "right hip",
    "right knee",
    "right knee",
    "right ankle-bone",
    "right ankle-bone",
    "foot tip right",
    "foot tip right",
]

DEFAULT_CODES = [
    "41001",
    "41001",
    "41000",
    "41000",
    "41002",
    "41002",
    "90001",
    "90001",
    "90002",
    "90002",
    "10000",
    "10000",
    "20000",
    "20000",
    "20100",
    "20100",
    "20300",
    "20300",
    "20311",
    "20311",
    "30000",
    "30000",
    "30100",
    "30100",
    "30300",
    "30300",
    "30311",
    "30311",
    "21000",
    "21000",
    "21100",
    "21100",
    "21200",
    "21200",
    "21301",
    "21301",
    "31000",
    "31000",
    "31100",
    "31100",
    "31200",
    "31200",
    "31301",
    "31301",
]


class SimiFloat:
    def __init__(self, value):
        if type(value) == SimiFloat:
            self.value = value
        elif type(value) == float:
            self.value = value
        else:
            self.value = float(str(value).replace(",", "."))

    def as_float(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return ("%.6f" % (self.value)).replace(".", ",")


header_template = """FileType	RawData
Version	150
Name	Raw data
Samples	%i
TimeOffset	%s
SamplesPerSecond	%s
Count	%i
"""


class SimiHeader:
    def __init__(self,
                 samples=0,
                 time_offset=SimiFloat(0.0),
                 samples_per_second=SimiFloat(240.0),
                 count=22):
        self.samples = samples
        self.time_offset = time_offset
        self.samples_per_second = samples_per_second

        # column count (x-y pairs)
        self.count = count

    @staticmethod
    def from_stream(fd):
        data = {}
        for line in fd:
            if line.strip() == "":
                break
            key, value = line.strip().split(DELIM)
            data[key] = value

        assert data.get("Version") == "150"
        samples = int(data["Samples"])
        time_offset = SimiFloat(data["TimeOffset"])
        samples_per_second = SimiFloat(data["SamplesPerSecond"])
        count = int(data["Count"])

        return SimiHeader(samples, time_offset, samples_per_second, count)

    def __str__(self) -> str:
        return header_template % (self.samples, self.time_offset, self.samples_per_second, self.count)


class SimiRowWrapper:
    def __init__(self, rows, cols, row_idx, ndims=2):
        self.rows = rows
        self.cols = cols
        self.row_idx = row_idx
        self.ndims = ndims

    def _colidx_by_name(self, col_name) -> int:
        idx = self.cols.index(col_name)
        if idx < 0:
            raise ValueError("Invalid column name: %s" % (col_name))
        return idx

    def __getitem__(self, col_name):
        idx = self._colidx_by_name(col_name)
        return self.rows[self.row_idx][idx:idx+self.ndims] 

    def __setitem__(self, idx, value):
        self.rows[self.row_idx][idx] = value

    def __str__(self):
        return DELIM.join([ '-' if r is None else str(r) for r in self.rows[self.row_idx] ])


class SimiData:
    def __init__(self, cols=DEFAULT_COLS, codes=DEFAULT_CODES, rows=[]) -> None:
        self.cols = cols
        self.codes = codes
        self.rows = rows

    def __str__(self) -> str:
        code_str = DELIM.join(self.codes)
        col_str = DELIM.join(self.cols)
        rows = "\n".join([DELIM.join(map(str, r)) for r in self.rows])
        return f"{code_str}\n{col_str}\n{rows}"

    def __len__(self) -> int:
        return len(self.rows)

    def get_row_wrapper(self, row_idx):
        return SimiRowWrapper(self.rows, self.cols, row_idx)

    @staticmethod
    def from_stream(fd):
        codes = next(fd).rstrip("\n").split(DELIM)
        cols = next(fd).rstrip("\n").split(DELIM)
        assert len(codes) == len(cols)
        rows = []
        for row in fd:
            raw_data = row.rstrip("\n").split(DELIM)
            rows.append([SimiFloat(r) if r != '' else None for r in raw_data])

        return SimiData(cols, codes, rows)

    def _build_column_lookup(self, cols):
        lookup = {}
        for i, col in enumerate(self.cols):
            if col not in lookup:
                lookup[col] = i
        return lookup
    
    def _get_empty_row(self):
        return [''] * len(self.cols)

    def add(self, obj):
        lookup = self._build_column_lookup(self.cols)
        row = self._get_empty_row()

        # populate columns
        for col, values in obj.items():
            idx_col = lookup[col]
            for idx_axis, value in enumerate(values):
                row[idx_col + idx_axis] = SimiFloat(value)

        self.rows.append(row)


class SimiDataFile:
    def __init__(self, header=SimiHeader(), data=SimiData()):
        self.header = header
        self.data = data

    def get_samples(self) -> int:
        return self.header.samples

    def get_time_offset(self) -> float:
        return self.header.time_offset.as_float()

    def get_samples_per_second(self) -> float:
        return self.header.samples_per_second.as_float()

    @staticmethod
    def from_file(path):
        fd = open(path)
        h = SimiHeader.from_stream(fd)
        data = SimiData.from_stream(fd)
        return SimiDataFile(h, data)

    def add_row(self, obj):
        self.data.add(obj)
        self.header.samples = len(self.data)

    def __getitem__(self, row_idx):
        return self.data.get_row_wrapper(row_idx)
    
    def __len__(self):
        return len(self.data.rows)

    def __str__(self) -> str:
        return "%s\n%s" % (self.header, self.data)
