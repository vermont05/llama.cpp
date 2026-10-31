import os
import glob
import shutil
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

cache_dir: str = ""


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


def make_server():
    srv = ServerPreset.tinyllama2()
    srv.n_slots = 1
    srv.n_predict = 4
    srv.temperature = 0.0
    srv.cache_ram = 1
    srv.debug = True
    srv.cache_disk = cache_dir
    fd, srv.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    return srv


@pytest.fixture(autouse=True)
def create_server():
    global server, cache_dir
    cache_dir = tempfile.mkdtemp(prefix='prompt-cache-disk-')
    server = make_server()
    yield
    server.stop()
    shutil.rmtree(cache_dir, ignore_errors=True)


def cache_files():
    return glob.glob(os.path.join(cache_dir, "pc-*.bin"))


# distinct prompts of about 150 tokens each, so that a few of them fill the 1 MiB RAM level
def make_prompt(i: int) -> str:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett"]
    return f"Story number {i}: " + " ".join(words[(i + k) % len(words)] + str(k * (i + 1)) for k in range(60))


def complete(prompt: str):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body


def test_restore_after_restart():
    global server
    server.start()

    prompt = make_prompt(0)
    first = complete(prompt)
    assert first["timings"]["prompt_n"] > 100

    # the slot is written to disk on exit
    server.stop()
    assert len(cache_files()) == 1

    server = make_server()
    server.start()
    log = LogReader(server.log_path)
    assert "disk prompt cache: 1 prompt(s)" in log.drain()

    res = complete(prompt)
    assert "disk prompt cache: read prompt" in log.drain()
    assert res["timings"]["cache_n"] > 0
    assert res["timings"]["prompt_n"] < first["timings"]["prompt_n"]
    assert res["content"] == first["content"]


def test_spill_on_eviction():
    global server
    server.start()
    log = LogReader(server.log_path)

    first = complete(make_prompt(0))

    # each new prompt moves the previous one into the RAM level, which evicts to disk once full
    for i in range(1, 12):
        complete(make_prompt(i))

    content = log.drain()
    assert "removing oldest entry" in content
    assert "disk prompt cache: queued prompt" in content or "disk prompt cache: wrote prompt" in content

    # the first prompt is gone from RAM, it comes back from disk (or from the write queue)
    res = complete(make_prompt(0))
    content = log.drain()
    assert "disk prompt cache: read prompt" in content or "disk prompt cache: taking back prompt" in content
    assert res["timings"]["cache_n"] > 0
    assert res["timings"]["prompt_n"] < first["timings"]["prompt_n"]
    assert res["content"] == first["content"]


def test_other_model_files_ignored():
    global server
    server.start()
    complete(make_prompt(0))
    server.stop()

    files = cache_files()
    assert len(files) == 1

    # a file of another fingerprint is neither indexed nor deleted
    other = os.path.join(cache_dir, "pc-0000000000000000-0000000000000000.bin")
    shutil.copy(files[0], other)

    server = make_server()
    server.start()
    log = LogReader(server.log_path)
    assert "disk prompt cache: 1 prompt(s)" in log.drain()
    assert os.path.exists(other)


def n_tokens_in_file(path: str) -> int:
    # header: u32 magic, u32 version, u64 fingerprint, u64 number of packed tokens
    with open(path, "rb") as f:
        f.seek(16)
        return int.from_bytes(f.read(8), "little")


def test_lru_order_kept_on_exit():
    global server
    server.start()

    # A, then B, then C (C stays in the slot, A and B go to the RAM level)
    for i, n in enumerate([20, 40, 60]):
        complete(make_prompt(i)[: 12 * n])
    server.stop()

    files = sorted(cache_files(), key=os.path.getmtime)
    assert len(files) == 3
    # the newest state (the slot) gets the newest mtime, so it is evicted last
    sizes = [n_tokens_in_file(f) for f in files]
    assert sizes == sorted(sizes)


def test_disk_size_zero_disables():
    global server
    server.cache_disk_size = 0
    server.start()
    log = LogReader(server.log_path)
    assert "disk prompt cache disabled by --cache-disk-size 0" in log.drain()
    complete(make_prompt(0))
    server.stop()
    assert len(cache_files()) == 0


def test_other_fingerprints_count_toward_limit():
    global server
    # an old file of another fingerprint, larger than the limit
    other = os.path.join(cache_dir, "pc-0000000000000000-0000000000000000.bin")
    with open(other, "wb") as f:
        f.write(b"\0" * (3 * 1024 * 1024))
    os.utime(other, (1, 1))

    server.cache_disk_size = 2
    server.start()
    log = LogReader(server.log_path)
    assert "removing least recently used file of another fingerprint" in log.drain()
    assert not os.path.exists(other)


def test_corrupt_file_removed():
    global server
    server.start()
    first = complete(make_prompt(0))
    server.stop()

    files = cache_files()
    assert len(files) == 1

    # cut the state, the header and tokens stay valid so the file is indexed
    size = os.path.getsize(files[0])
    with open(files[0], "r+b") as f:
        f.truncate(size // 2)

    server = make_server()
    server.start()
    log = LogReader(server.log_path)

    res = complete(make_prompt(0))
    assert "is corrupt, removing it" in log.drain()
    assert res["content"] == first["content"]
    assert not os.path.exists(files[0])


def test_cached_prompt_beats_slot():
    global server
    server.cache_ram = 100
    server.start()
    log = LogReader(server.log_path)

    def n_tokens(text: str) -> int:
        res = server.make_request("POST", "/tokenize", data={"content": text})
        return len(res.body["tokens"])

    system = "You are a helpful assistant who answers questions about stories, places and people. " * 2
    story = " ".join(f"word{k}" for k in range(66))

    # the slot is picked by similarity (> 0.1), and the cache gains more than 256 tokens over it
    n_system = n_tokens(system)
    n_story  = n_tokens(system + story) - n_system
    assert n_system > 0.1 * (n_system + n_story + 8)
    assert n_story > 300
    assert n_system + n_story + 16 < 512

    # conversation A fills the slot
    a = complete(system + story)

    # conversation B shares only the system prompt and keeps most of the slot, A moves to the RAM cache
    complete(system + "Hello there")

    # A continues: the slot shares only the system prompt, the cache holds all of A
    res = complete(system + story + " The end.")
    assert "prompt cache shares a longer prefix with the task" in log.drain()
    assert res["timings"]["cache_n"] >= a["timings"]["prompt_n"] - 8
    assert res["timings"]["prompt_n"] < 32
