from __future__ import annotations

import heapq
import io
import os
from collections import Counter, deque
from functools import total_ordering
from math import ceil
from os import PathLike
from pathlib import Path

import tqdm


@total_ordering
class CodeNode:
    def __init__(self, symbol: int | None, weight: int) -> None:
        self.symbol = symbol
        self.weight = weight
        self.left: CodeNode | None = None
        self.right: CodeNode | None = None

    def set_left(self, n: CodeNode) -> None:
        self.left = n

    def set_right(self, n: CodeNode) -> None:
        self.right = n

    def set_branch(self, seq: str, symbol: int) -> CodeNode:
        if seq[0] == "0":
            if not self.left:
                self.left = CodeNode(None, -1)
            if len(seq) != 1:
                self.left.set_branch(seq[1:], symbol)
            else:
                self.left.symbol = symbol

        else:
            if not self.right:
                self.right = CodeNode(None, -1)
            if len(seq) != 1:
                self.right.set_branch(seq[1:], symbol)
            else:
                self.right.symbol = symbol

    def __eq__(self, other: CodeNode) -> bool:
        return self.weight == other.weight

    def __lt__(self, other: CodeNode) -> bool:
        return self.weight < other.weight


def get_file_freqs(file: Path) -> dict[int, int]:
    freqs: dict[int, int] = {}
    with open(file, mode="rb") as fin:
        for byte in iter(lambda: fin.read(1), b""):
            k = ord(byte)
            freqs[k] = freqs.setdefault(k, 0) + 1
    return freqs


def get_codes(freqs: dict[int, int]) -> dict[int, str]:
    h = [CodeNode(k, v) for k, v in freqs.items()]
    heapq.heapify(h)

    while len(h) >= 2:
        n1 = heapq.heappop(h)
        n2 = heapq.heappop(h)
        n = CodeNode(None, n1.weight + n2.weight)
        n.set_left(n1)
        n.set_right(n2)
        heapq.heappush(h, n)

    root_n = h[0]

    iter_que: deque[tuple[CodeNode, str]] = deque()
    iter_que.append((root_n, ""))
    code_map: dict[int, str] = {}

    while len(iter_que):
        n, seq = iter_que.pop()

        if n.symbol is not None:
            code_map[n.symbol] = seq
            continue

        if n.right is not None:
            iter_que.append((n.right, seq + "1"))
        if n.left is not None:
            iter_que.append((n.left, seq + "0"))

    return code_map


def normalize_codes(code_map: dict[int, str]) -> dict[int, str]:
    srt_pairs = sorted(code_map.items(), key=lambda x: len(x[1]), reverse=False)
    len_pairs = [(k, len(v)) for k, v in srt_pairs]
    pairs: list[tuple[int, int, int]] = []

    i = len_pairs.pop(0)
    symbol = i[0]
    val = 0
    code_len = i[1]
    pairs.append((symbol, val, code_len))

    for symbol, code_len in len_pairs:
        pre_val = pairs[-1][1]
        pre_len = pairs[-1][2]

        if code_len == pre_len:
            pairs.append((symbol, pre_val + 1, code_len))
        else:
            pairs.append((symbol, (pre_val + 1) << (code_len - pre_len), code_len))

    seq_pairs: dict[int, str] = {}
    for symbol, val, code_len in pairs:
        bin_str = bin(val)[2:]
        assert len(bin_str) <= code_len
        bin_str = bin_str.rjust(code_len, "0")
        seq_pairs[symbol] = bin_str

    return seq_pairs


def encode(pre_str: str, block: bytes, map: dict[int, str]) -> tuple[bytes, str]:
    bufs = [pre_str]
    for k in block:
        bufs.append(map[k])
    buf = "".join(bufs)
    end_pos = 8 * (len(buf) // 8) - 1
    block_bytes = bytes(int(buf[i : i + 8], base=2) for i in range(0, end_pos + 1, 8))
    return block_bytes, buf[end_pos + 1 :]


def calc_head(code_map: dict[int, str]) -> bytes:
    max_code_len = max(len(code) for code in code_map.values())
    counts = dict(Counter([len(code) for code in code_map.values()]))
    for i in range(1, max_code_len + 1):
        counts.setdefault(i, 0)
    srt_counts = sorted(counts.items(), key=lambda x: x[0], reverse=False)

    lens = [freq for _, freq in srt_counts]
    symbols = list(code_map.keys())

    head = [0, len(lens), len(symbols)]
    head.extend(lens)
    head.extend(symbols)
    head.insert(0, len(head))
    return bytes(head)


def compress(
    file: str | PathLike[str] | Path, out_file: str | PathLike[str] | Path
) -> None:
    f = Path(file).resolve(strict=True)
    out = Path(out_file).resolve()

    freqs = get_file_freqs(file)
    code_map = normalize_codes(get_codes(freqs))
    head = calc_head(code_map)

    with open(f, "rb") as fin, open(out, "wb") as fout:
        fout.write(head)
        remain = ""

        progress = tqdm.tqdm(total=1000)
        fsize = os.path.getsize(f)
        step = ceil(fsize / 1024 / 1000)

        for block in iter(lambda: fin.read(1024), b""):
            block_bytes, remain = encode(remain, block, code_map)
            fout.write(block_bytes)
            progress.update(step)

        last_byte = bytes([int(remain.ljust(8, "0"), base=2)])
        fout.write(last_byte)
        fout.seek(1, os.SEEK_SET)
        fout.write(bytes([8 - len(remain)]))
        progress.close()


def decompress(
    file: str | PathLike[str] | Path, out_file: str | PathLike[str] | Path
) -> None:
    f = Path(file).resolve(strict=True)
    out = Path(out_file).resolve()

    with open(f, "rb") as fin, open(out, "wb") as fout:
        head_shift = ord(fin.read(1))
        lookup_map, pad_len = parse_head(bytes([head_shift]) + fin.read(head_shift))
        root = CodeNode(None, -1)
        for k, v in lookup_map.items():
            root.set_branch(k, v)

        cur = root
        symbols: list[int] = []
        progress = tqdm.tqdm(total=1000)
        fsize = os.path.getsize(f)
        step = int(fsize / 1000) / 2

        for b_step, (b1, b2) in enumerate(
            iter(lambda: (fin.read(1), fin.read(1)), (b"", b""))
        ):
            if b2 != b"":
                val = bin(ord(b1))[2:].rjust(8, "0") + bin(ord(b2))[2:].rjust(8, "0")
            else:
                val = bin(ord(b1))[2:][:-pad_len]

            for c in val:
                if c == "0":
                    cur = cur.left
                else:
                    cur = cur.right

                if cur.symbol is not None:
                    symbols.append(cur.symbol)
                    cur = root

            if len(symbols) >= 1024:
                fout.write(bytes(symbols))
                symbols.clear()

            if b_step % step == 0:
                progress.update(1)

        if len(symbols):
            fout.write(bytes(symbols))
            symbols.clear()

        progress.close()


def parse_head(head: bytes) -> tuple[dict[str, int], int]:
    padding = head[1]
    freq_block_len = head[2]

    freqs = [head[i] for i in range(4, freq_block_len + 4)]
    symbols = deque(head[i] for i in range(4 + freq_block_len, len(head)))

    len_pairs: list[tuple[int, int]] = []
    for bit_len, freq in enumerate(freqs, 1):
        if freq != 0:
            for i in range(freq):
                len_pairs.append((symbols.popleft(), bit_len))

    pairs: list[tuple[int, int, int]] = []

    i = len_pairs.pop(0)
    symbol = i[0]
    val = 0
    code_len = i[1]
    pairs.append((symbol, val, code_len))

    for symbol, code_len in len_pairs:
        pre_val = pairs[-1][1]
        pre_len = pairs[-1][2]

        if code_len == pre_len:
            pairs.append((symbol, pre_val + 1, code_len))
        else:
            pairs.append((symbol, (pre_val + 1) << (code_len - pre_len), code_len))

    seq_pairs: dict[int, str] = {}
    for symbol, val, code_len in pairs:
        bin_str = bin(val)[2:]
        assert len(bin_str) <= code_len
        bin_str = bin_str.rjust(code_len, "0")
        seq_pairs[symbol] = bin_str

    lookup_map = {v: k for k, v in seq_pairs.items()}
    return lookup_map, padding


compress("test.rar", "test.chc")
# decompress("out.chc", "out.txt")
