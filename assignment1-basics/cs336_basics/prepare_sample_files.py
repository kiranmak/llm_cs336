import os
import sys
import random
import pathlib

# Add the project root directory to python's import path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data"
def get_random_access_block(file_path, keyword):
    keyword_bytes = keyword.encode("utf-8")

    def _find_next_keyword(f):
        buffer = b""
        chunk_size = 8192
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return None
            buffer += chunk
            idx = buffer.find(keyword_bytes)
            if idx != -1:
                return f.tell() - len(buffer) + idx
            if len(buffer) > len(keyword_bytes):
                buffer = buffer[-len(keyword_bytes):]

    def _read_until_next_keyword(f):
        result = bytearray(keyword_bytes)
        buffer = b""
        chunk_size = 8192
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                result.extend(buffer)
                return bytes(result)
            buffer += chunk
            idx = buffer.find(keyword_bytes)
            if idx != -1:
                result.extend(buffer[:idx])
                f.seek(f.tell() - len(buffer) + idx)
                return bytes(result)
            if len(buffer) > len(keyword_bytes):
                result.extend(buffer[:-len(keyword_bytes)])
                buffer = buffer[-len(keyword_bytes):]

    blocks = []
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return None

    with open(file_path, "rb") as f:
        for _ in range(10):
            start_pos = random.randrange(file_size)
            f.seek(start_pos)

            keyword_pos = _find_next_keyword(f)
            if keyword_pos is None:
                continue
            f.seek(keyword_pos + len(keyword_bytes))
            blocks.append(_read_until_next_keyword(f))

    if not blocks:
        return None

    return random.choice(blocks).decode("utf-8", errors="replace")

# Example usage
input_files = ["TinyStoriesV2-GPT4-train.txt","owt_train.txt"]
output_files = ["TinyStoriesV2-GPT4-samples.txt","owt_samples.txt"]
KEYWORD = "<|endoftext|>"
for i in range(len(input_files)):
    in_file=DATA_PATH / input_files[i]
    out_file=DATA_PATH / output_files[i]
    result = get_random_access_block(in_file, KEYWORD)
    with open(out_file, "w") as f:
        f.write(result)
    f.close()
