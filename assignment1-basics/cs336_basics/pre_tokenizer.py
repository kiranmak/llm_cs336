import os
import time, datetime # time_perf
import csv, pickle # for writing output file
from typing import BinaryIO
from itertools import islice # from printing Counter
import regex as re
import mmap
from multiprocessing import current_process
import multiprocessing
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        return tuple(text.encode("utf-8"))

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

    def log_pid(self, start, end):
        pid = os.getpid()
        name = current_process().name
        # Divide by 1024 twice to get Megabytes
        mb = (end - start) / (1024 * 1024)
        print(f"{name} (PID: {pid}), {mb:.2f} MB; start {start} end {end}")

    def pre_tokenize_chunk(self, args) -> Counter:
        """
         Worker function made standalone so it can be cleanly picked 
         and distributed across Ubuntu processes without serialization errors.
         """
        fname, start, end, special_tokens, special_split_regex, bpe_regex, set_key = args

        self.log_pid(start, end)
        match_details: Counter = Counter()

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


    def pre_tokenize_file(self, fname: str, encode_it: bool):
        start_time = time.perf_counter()
        num_workers = 12
        pre_tokens: Counter = Counter()

        with open(fname, "rb") as f:
            boundaries = find_chunk_boundaries( f, num_workers, b"<|endoftext|>")

        set_key = self.pretoken_bkey if encode_it else self.pretoken_key
        # split work
        ctx = multiprocessing.get_context('spawn')

        print(f"Launching pre-tokenization across {num_workers} parallel CPU workers...")

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            # Map submits all tasks simultaneously and yields results as they finish
            futures = []
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                task_args = (
                    fname,
                    start,
                    end,
                    self.special_tokens,
                    self.special_split_regex,
                    self.bpe_regex,
                    set_key
             )
            futures.append(executor.submit(self.pre_tokenize_chunk, task_args))

            # 2. Process results on-the-fly as they complete
            for idx, future in enumerate(as_completed(futures)):
                chunk_counter = future.result()  # Fetch one counter

                print(f" -> Worker chunk completed. Merging into global vocabulary...")
                pre_tokens += chunk_counter     # Aggregate immediately

                # 3. CRITICAL: Delete the reference immediately so garbage collection
                # can free the worker's memory allocation right away
                del chunk_counter
                print(f" -> Worker chunk successfully merged. ({idx + 1}/{num_workers})",
                      flush=True)

            # --- THE HARD BARRIER ---
            print("\n[System] Finished aggregating final vocabulary structures...",
                  flush=True)
        end_time = time.perf_counter()
        print_time("PRETOKEN", end_time - start_time)
        time.sleep(1)

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
    fname = "./data/owt_samples.txt"
    pre_tokens = pretokenizer.pre_tokenize_file(
                            fname, encode_it=True)
    #pretokenizer.show_pre_tokens(pre_tokens)
    pretokenizer.write_pre_tokens(pre_tokens, ftype="pkl", fname=fname)
