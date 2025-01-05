from __future__ import annotations

import heapq
import io
import os
import shutil
import struct
from collections import Counter, deque
from functools import total_ordering
from math import ceil
from os import PathLike
from pathlib import Path
from time import perf_counter

import tqdm


@total_ordering
class CodeNode:
    """
    Huffman 编码树的结点
    """

    def __init__(self, symbol: int | None, weight: int) -> None:
        self.symbol = symbol
        self.weight = weight
        self.left: CodeNode | None = None
        self.right: CodeNode | None = None

    def set_left(self, n: CodeNode) -> None:
        """
        设置左子结点

        :param: n: 左子结点
        """
        self.left = n

    def set_right(self, n: CodeNode) -> None:
        """
        设置右子结点

        :param: n: 右子结点
        """
        self.right = n

    def set_branch(self, seq: str, symbol: int) -> None:
        """
        递归建立分支，这将用于解码时，从译码表到码树的恢复

        :param seq: 译码表中的码字，例如 "00001"
        :param symbol: 对应的符号，这里将是一个字节值（0-255 之间的值）
        """
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
    """
    获取文件中的每个符号（也就是单个字节）的频率

    :param file: 文件路径
    :return: 频率信息
    """
    freqs: dict[int, int] = {}
    with open(file, mode="rb") as fin:
        for byte in iter(lambda: fin.read(1), b""):
            k = ord(byte)
            freqs[k] = freqs.setdefault(k, 0) + 1
    return freqs


def get_codes(freqs: dict[int, int]) -> dict[int, str]:
    """
    依据提供的频率信息，建立编码表

    :param freqs: 频率信息
    :return: 编码表
    """

    # 从结点权重小的开始建立树
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

    # 遍历树，建立编码表
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
    """
    规范化编码表为“范式哈夫曼编码”

    :param code_map: 编码表
    :return: 规范化后的编码表
    """
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


def encode_block(remain: str, block: bytes, map: dict[int, str]) -> tuple[bytes, str]:
    """
    逐块编码

    :param pre_str: 上一轮逐块编码剩余的字节
    :param block: 此轮编码的块
    :param map: 编码表
    :return: 二元组，第一个元素是编码后的块，第二个元素是此轮编码剩余的字节
    """
    buf_seq = [remain]
    for k in block:
        buf_seq.append(map[k])
    buf = "".join(buf_seq)
    end_pos = 8 * (len(buf) // 8) - 1
    block_bytes = bytes(int(buf[i : i + 8], base=2) for i in range(0, end_pos + 1, 8))
    return block_bytes, buf[end_pos + 1 :]


def calc_head(code_map: dict[int, str]) -> bytes:
    """计算编码后文件的头部信息

    :param code_map: 编码表
    :return: 头部信息块
    """
    # 头部信息块的格式：
    # [编码表码字长度信息, 符号个数溢出标志, 文件中拥有的符号个数, 编码表码字长度序列, 编码表符号序列]
    #
    # 但实际上，由于不等长编码的特性，最后编码完成后不一定是整字节的，因此还需要填充，因此头部信息块还需要提供填充长度信息
    # 在本函数外，将会添加这一填充信息。因此最终的头部信息块格式：
    #
    # [填充长度, 编码表码字长度信息, 符号个数溢出标志, 文件中拥有的符号个数, 编码表码字长度序列, 编码表符号序列]
    max_code_len = max(len(code) for code in code_map.values())
    counts = dict(Counter([len(code) for code in code_map.values()]))
    for i in range(1, max_code_len + 1):
        counts.setdefault(i, 0)
    srt_counts = sorted(counts.items(), key=lambda x: x[0], reverse=False)

    freq_lens = [freq for _, freq in srt_counts]
    symbols = list(code_map.keys())
    of_flag = 0
    if len(symbols) == 256:
        of_flag = 1

    head = [len(freq_lens), of_flag, len(symbols) if not of_flag else 0]
    head.extend(freq_lens)
    head.extend(symbols)
    len_bytes = struct.pack("<H", len(head))
    return bytes([0]) + len_bytes + bytes(head)


def parse_head(fin: io.BufferedReader) -> tuple[dict[str, int], int]:
    """解析文件头部信息

    :param fin: 文件流
    :return: 二元组，第一个元素是译码表，第二个元素是填充长度
    """
    # 头部信息块格式：
    # [填充长度, 编码表码字长度信息, 符号个数溢出标志, 文件中拥有的符号个数, 编码表码字长度序列, 编码表符号序列]
    pad_len = ord(fin.read(1))
    head_shift = struct.unpack("<H", fin.read(2))[0]
    block = fin.read(head_shift)

    freq_len = block[0]
    of_flag = block[1]
    if of_flag:
        symbol_len = 256
    else:
        symbol_len = block[2]

    freqs = [block[i] for i in range(3, 3 + freq_len)]
    symbols = deque(block[i] for i in range(3 + freq_len, 3 + freq_len + symbol_len))

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
    return lookup_map, pad_len


def encode(
    file: str | PathLike[str] | Path, out_file: str | PathLike[str] | Path
) -> None:
    """进行 Huffman 编码

    :param file: 需要编码的文件路径
    :param out_file: 编码后的文件保存路径
    """
    f = Path(file).resolve(strict=True)
    out = Path(out_file).resolve()

    # 计算频率，建立编码表，规范化编码表
    freqs = get_file_freqs(file)
    code_map = normalize_codes(get_codes(freqs))
    head = calc_head(code_map)

    with open(f, "rb") as fin, open(out, "wb") as fout:
        # 写入头部信息
        fout.write(head)
        remain = ""

        progress = tqdm.tqdm(total=1000)
        fsize = os.path.getsize(f)
        step = ceil(fsize / 1024 / 1000)

        # 逐块编码（1KB 为一块）
        for block in iter(lambda: fin.read(1024), b""):
            block_bytes, remain = encode_block(remain, block, code_map)
            fout.write(block_bytes)
            progress.update(step)

        # 填充至整字节，调整文件指针，在头部写入填充长度信息
        last_byte = bytes([int(remain.ljust(8, "0"), base=2)])
        fout.write(last_byte)
        fout.seek(0, os.SEEK_SET)
        fout.write(bytes([8 - len(remain)]))
        progress.close()


def decode(
    file: str | PathLike[str] | Path, out_file: str | PathLike[str] | Path
) -> None:
    """进行 Huffman 解码

    :param file: 需要解码的，编码后的文件路径
    :param out_file: 解码输出的文件保存路径
    """
    f = Path(file).resolve(strict=True)
    out = Path(out_file).resolve()

    with open(f, "rb") as fin, open(out, "wb") as fout:
        # 解析头部信息，获得译码表和末字节填充长度
        lookup_map, pad_len = parse_head(fin)

        # 从译码表建立码树
        root = CodeNode(None, -1)
        for k, v in lookup_map.items():
            root.set_branch(k, v)

        # 初始化译码指针和译码结果缓存
        cur = root
        symbols: list[int] = []
        # 初始化进度条
        progress = tqdm.tqdm(total=1000)
        fsize = os.path.getsize(f)
        step = ceil(fsize / 1024 / 1000 / 2)

        # 两个字节为一块，逐块解码
        # 不足两字节时，后一字节为空，此时我们就知道第一字节为末字节
        for b_step, (b1, b2) in enumerate(
            iter(lambda: (fin.read(1), fin.read(1)), (b"", b""))
        ):
            if b2 != b"":
                val = bin(ord(b1))[2:].rjust(8, "0") + bin(ord(b2))[2:].rjust(8, "0")
            else:
                val = bin(ord(b1))[2:][:-pad_len]

            # 遍历码树，串行解码
            for c in val:
                if c == "0":
                    cur = cur.left
                else:
                    cur = cur.right

                # 抵达码树叶结点，即找到一个符号
                if cur.symbol is not None:
                    symbols.append(cur.symbol)
                    cur = root

            # 缓存解码结果，每 1024 个符号（1KB）写入一次文件
            if len(symbols) >= 1024:
                fout.write(bytes(symbols))
                symbols.clear()

            if b_step % step == 0:
                progress.update(1)

        # 遍历结束时，缓存中的内容也记得写入文件
        if len(symbols):
            fout.write(bytes(symbols))
            symbols.clear()

        progress.close()


def test(file: str, huf_file: str, out_flie: str) -> None:
    """测试函数

    :param file: 文件路径
    :param huf_file: 编码生成文件的路径
    :param out_flie: 解码生成文件的路径
    """
    fname = file
    huf_fname = huf_file
    out_fname = out_flie

    print(f"开始编码 {fname}")
    start = perf_counter()
    encode(fname, huf_fname)
    duration = perf_counter() - start

    fsize1 = os.path.getsize(fname)
    fsize2 = os.path.getsize(huf_fname)
    print(
        f"编码完成，压缩率：{(fsize1-fsize2)/fsize1:.2f}，"
        f"耗时：{duration:.2f} 秒，"
        f"速率：{fsize1 / duration / 1024:.2f} KB/s"
    )

    print(f"开始解码 {fname}")
    start = perf_counter()
    decode(huf_fname, out_fname)
    duration = perf_counter() - start

    fsize1 = os.path.getsize(huf_fname)
    fsize2 = os.path.getsize(out_fname)
    print(
        f"解码完成，耗时：{duration:.2f} 秒，"
        f"速率：{fsize1 / duration / 1024:.2f} KB/s"
    )
    print()


if __name__ == "__main__":
    os.chdir(str(Path(__file__).parent.resolve()))
    try:
        shutil.rmtree("huf")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree("out")
    except FileNotFoundError:
        pass

    os.mkdir("huf")
    os.mkdir("out")

    test("test.txt", "./huf/txt.huf", "./out/out.txt")
    test("test.bmp", "./huf/bmp.huf", "./out/out.bmp")
    test("test.wav", "./huf/wav.huf", "./out/out.wav")
