import os
import time, datetime # time_perf
import csv, pickle # for writing output file
from typing import BinaryIO
from itertools import islice # from printing Counter
import regex as re
import mmap
from multiprocessing import Pool, current_process
from collections import Counter, defaultdict

OUT= "./out"
IN= "./data"

def print_time(fn, elapsed):
    hr, remainder  = divmod(elapsed, 3600)
    min, remainder = divmod(remainder, 60)
    sec  = int(remainder)
    msec = int((remainder - sec) * 1000)
    print(f"Elapsed Time in {fn}: {int(hr):02}:{int(min):02}:{sec:02}.{msec:03}")

def load_pkl(fname):
    ## Reload later
    with open(fname, 'rb') as f:
        pre_tokens = pickle.load(f)
    return pre_tokens

# {Raw Text} --> {Protect Special Tokens} --> {Regex Split} -->
#   --> {Space-to-Unicode Mapping} --> {Byte Conversion}

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class BPEPreTokenizer:
    def __init__(self, special_tokens: list[str]):
        self.special_tokens = special_tokens

        escaped_tokens = [re.escape(t) for t in special_tokens]
        self.special_split_regex = re.compile(f"({'|'.join(escaped_tokens)})")

        self.bpe_regex = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @staticmethod
    def pretoken_key(text: str) -> tuple[str, ...]:
        """One Counter key per pretoken: tuple of single-character strings."""
        return tuple(text)

    @staticmethod
    def pretoken_bkey(text: str) -> tuple[[int], ...]:
        """One Counter key per pretoken: tuple of single-character strings."""
        result = list(map(int, text.encode("utf-8")))
        return tuple(result)

    def pre_tokenize_str(self, raw_string: str) -> Counter:
        match_details: Counter = Counter()

        # If raw_string is a string, we split it directly
        pieces = self.special_split_regex.split(raw_string)

        for piece in pieces:
            if not piece:
                continue

            if piece in self.special_tokens:
                match_details[self.pretoken_key(piece)] += 1
            else:
                for match in self.bpe_regex.finditer(piece):
                    match_details[self.pretoken_key(match.group())] += 1

        return match_details

    def log_pid(self, x, start, end):
        pid = os.getpid()
        name = current_process().name
        print(f"Task {x} {name} (PID: {pid}), start {start} end {end}")

    def pre_tokenize_chunk(self, args) -> Counter:
        fname, inst, encode_it, start, end = args

        match_details: Counter = Counter()
        if encode_it:
            set_key = self.pretoken_bkey
        else:
            set_key = self.pretoken_key

        # Open the file and memory-map it INSIDE the worker
        with open(fname, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                chunk = mm[start:end].decode("utf-8")
                pieces = self.special_split_regex.split(chunk)
                for piece in pieces:
                    if not piece or piece in self.special_tokens:
                        continue
                    for match in self.bpe_regex.finditer(piece):
                        match_details[set_key(match.group())] += 1
        return match_details


    def pre_tokenize_file(self, fname: str, encode_it:bool):
        start_time = time.perf_counter()  # Before
        num_processes = 4
        with open(fname, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes,
                                                b"<|endoftext|>")
        f.close()
        args = []

        for i in range(len(boundaries) - 1):
            arg = (fname, i, encode_it,
                   boundaries[i], boundaries[i+1])
            args.append(arg)

        with Pool() as pool:
            results = pool.map(self.pre_tokenize_chunk, args)
            pre_tokens = sum(results, Counter())

        end_time = time.perf_counter()    # After
        print_time("PRE_TOKENIZE", end_time - start_time)

        return pre_tokens

    def show_pre_tokens(self, pre_tokens:Counter):
        # Get the first 10 inserted elements
        first_10 = islice(pre_tokens.items(), 10)
        for item, count in first_10:
            print(f"{item}: {count}")

    def write_pre_tokens(self, pre_tokens, ftype:str, fname: str):

        from pathlib import Path
        path_obj = Path(fname)
        out_fname = Path(OUT) / f"{path_obj.stem}.csv"
        if ftype in ("csv", "both"):
            out_fname.parent.mkdir(parents=True, exist_ok=True)
            with open(out_fname, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Item', 'Count']) # Header
                writer.writerows(pre_tokens.items())

        if ftype in ("pkl", "both"):
            # Create a new filename based on the input
            out_fname = Path(OUT) / f"{path_obj.stem}.pkl"
            out_fname.parent.mkdir(parents=True, exist_ok=True)
            with open(out_fname, 'wb') as f:
                pickle.dump(pre_tokens, f)
        return out_fname

# --- Test it out ---
if __name__ == "__main__":
    pretokenizer = BPEPreTokenizer(special_tokens=["<|endoftext|>"])
    # run with file
    fname = "./data/test_samples.txt"
    pre_tokens = pretokenizer.pre_tokenize_file(
                            fname, encode_it=True)
    pretokenizer.show_pre_tokens(pre_tokens)
