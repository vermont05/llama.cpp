#include "server-cache-disk.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static constexpr uint32_t DISK_MAGIC   = 0x44435053; // "SPCD"
static constexpr uint32_t DISK_VERSION = 2; // 2: checkpoint flags and positions, root entries

// blobs are written in chunks, so that fetch() can stop a write early
static constexpr size_t DISK_CHUNK = 64ull*1024*1024;

// free space the disk level leaves on the file system: 5% of it, at least 1 GiB
static size_t disk_reserve(const fs::space_info & space) {
    return std::max<size_t>(1024ull*1024*1024, space.capacity / 20);
}

static constexpr double MiB = 1024.0*1024.0;

static uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const auto * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

namespace {

// FILE wrapper that counts bytes and remembers the first error
struct disk_file {
    FILE * f  = nullptr;
    bool   ok = false;
    size_t n  = 0; // bytes read or written

    disk_file(const std::string & path, const char * mode) : f(ggml_fopen(path.c_str(), mode)), ok(f != nullptr) {}

    ~disk_file() {
        if (f) {
            fclose(f);
        }
    }

    void write(const void * src, size_t size) {
        if (ok && size > 0) {
            ok = fwrite(src, 1, size, f) == size;
            n += size;
        }
    }

    void read(void * dst, size_t size) {
        if (ok && size > 0) {
            ok = fread(dst, 1, size, f) == size;
            n += size;
        }
    }

    template <typename T> void write_val(const T & val) {
        write(&val, sizeof(val));
    }

    template <typename T> T read_val() {
        T val{};
        read(&val, sizeof(val));
        return val;
    }

    bool seek(uint64_t offset) {
#if defined(_WIN32)
        return _fseeki64(f, (__int64) offset, SEEK_SET) == 0;
#else
        return fseeko(f, (off_t) offset, SEEK_SET) == 0;
#endif
    }

    // flush to the device, so that a renamed file is complete after a crash
    bool sync() {
        if (!ok || fflush(f) != 0) {
            return false;
        }
#if defined(_WIN32)
        return _commit(_fileno(f)) == 0;
#else
        return fsync(fileno(f)) == 0;
#endif
    }
};

}

static bool write_blob(disk_file & f, const std::vector<uint8_t> & buf, const std::atomic<bool> & cancel) {
    f.write_val<uint64_t>(buf.size());
    for (size_t off = 0; off < buf.size() && f.ok; off += DISK_CHUNK) {
        if (cancel.load()) {
            return false;
        }
        f.write(buf.data() + off, std::min(DISK_CHUNK, buf.size() - off));
    }
    return f.ok;
}

static bool read_blob(disk_file & f, std::vector<uint8_t> & buf, size_t file_size) {
    const uint64_t size = f.read_val<uint64_t>();
    // check the size against the bytes left before allocating, the field may be corrupted
    if (!f.ok || size > file_size - f.n) {
        return false;
    }
    buf.resize(size);
    f.read(buf.data(), size);
    return f.ok;
}

static std::string hex64(uint64_t val) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016" PRIx64, val);
    return buf;
}

// a path as UTF-8 (std::filesystem::path::u8string() is std::u8string since C++20)
static std::string fs_path_to_utf8(const fs::path & path) {
    const auto res = path.u8string();
    return std::string(res.begin(), res.end());
}

static std::vector<int64_t> ckpt_positions(const server_prompt & prompt) {
    std::vector<int64_t> res;
    for (const auto & ckpt : prompt.checkpoints) {
        res.push_back(ckpt.n_tokens);
    }
    return res;
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

//
// server_prompt_cache_disk
//

server_prompt_cache_disk::server_prompt_cache_disk(const std::string & dir, size_t limit_size, size_t limit_pending, uint64_t fingerprint, bool has_mtmd)
    : dir(dir), prefix("pc-" + hex64(fingerprint) + "-"), limit_size(limit_size), limit_pending(limit_pending), fingerprint(fingerprint), has_mtmd(has_mtmd) {
    if (const char * ttl = std::getenv("LLAMA_CACHE_DISK_TTL")) {
        ttl_hours = std::max(0.0, std::atof(ttl));
    }
    if (const char * ttl = std::getenv("LLAMA_CACHE_DISK_ROOT_TTL")) {
        root_ttl_hours = std::max(0.0, std::atof(ttl));
    }
    if (ttl_hours > 0 || root_ttl_hours > 0) {
        SRV_INF("disk prompt cache: entries expire after %.1f h unused, root entries after %.1f h (0 = never); roots are evicted last "
                "while they hold at most a quarter of the limit\n", ttl_hours, root_ttl_hours);
    }
    scan();

    worker = std::thread([this]() { worker_loop(); });
}

server_prompt_cache_disk::~server_prompt_cache_disk() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        stopping = true;

        if (!queue.empty()) {
            SRV_INF("disk prompt cache: writing %zu remaining prompt(s) (%.3f MiB) before exit\n", queue.size(), size_pending / MiB);
        }
    }
    cv.notify_all();

    if (worker.joinable()) {
        worker.join();
    }
}

std::list<server_prompt_cache_disk::entry>::iterator server_prompt_cache_disk::find_entry(int64_t id) {
    return std::find_if(entries.begin(), entries.end(), [id](const entry & e) { return e.id == id; });
}

size_t server_prompt_cache_disk::size_on_disk() const {
    size_t res = 0;
    for (const auto & e : entries) {
        if (e.on_disk) {
            res += e.size;
        }
    }
    return res;
}

size_t server_prompt_cache_disk::size() const {
    std::lock_guard<std::mutex> lk(mtx);
    return size_on_disk();
}

size_t server_prompt_cache_disk::n_entries() const {
    std::lock_guard<std::mutex> lk(mtx);
    return std::count_if(entries.begin(), entries.end(), [](const entry & e) { return !e.foreign; });
}

void server_prompt_cache_disk::erase_entry(std::list<entry>::iterator it) {
    GGML_ASSERT(!it->busy());

    if (it->on_disk) {
        std::error_code ec;
        fs::remove(fs::u8path(it->path), ec);
    }

    if (it->pending) {
        size_pending -= it->size_mem;
        it->pending.reset();
    }

    queue.erase(std::remove(queue.begin(), queue.end(), it->id), queue.end());

    entries.erase(it);
}

bool server_prompt_cache_disk::covers(const server_tokens & a_tokens, const std::vector<int64_t> & a_ckpts,
                                      const server_tokens & b_tokens, const std::vector<int64_t> & b_ckpts) {
    if (a_tokens.size() < b_tokens.size() || a_tokens.get_common_prefix(b_tokens) != b_tokens.size()) {
        return false;
    }

    // a memory that rolls back only to checkpoints resumes b's prefix at b's checkpoints, a continuation without them
    // does not replace it (b's own end is one of them for a root entry)
    for (const int64_t pos : b_ckpts) {
        if (std::find(a_ckpts.begin(), a_ckpts.end(), pos) == a_ckpts.end()) {
            return false;
        }
    }

    return true;
}

void server_prompt_cache_disk::drop_superseded(const entry & cur) {
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);

        // a waiting entry is left alone: fetch() may wait for it after a cancelled write
        if (&*it != &cur && !it->busy() && !it->pending && !it->foreign && (!it->root || cur.root) &&
                covers(cur.tokens, cur.ckpts, it->tokens, it->ckpts)) {
            SRV_TRC("disk prompt cache: removing prompt with %zu tokens, contained in the new one\n", it->tokens.size());
            erase_entry(it);
        }

        it = next;
    }
}

void server_prompt_cache_disk::reap_expired(int64_t id_keep) {
    if (ttl_hours <= 0 && root_ttl_hours <= 0) {
        return;
    }

    const auto now     = time_point::clock::now();
    const auto hours   = [](double h) { return std::chrono::duration_cast<time_point::duration>(std::chrono::duration<double, std::ratio<3600>>(h)); };
    const auto cut     = ttl_hours      > 0 ? now - hours(ttl_hours)      : time_point::min();
    const auto cut_rt  = root_ttl_hours > 0 ? now - hours(root_ttl_hours) : time_point::min();

    size_t n_removed = 0, n_roots = 0;
    size_t n_bytes   = 0;
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);
        // conversation entries and files of other fingerprints use the short TTL, root entries the long one
        if (it->on_disk && !it->busy() && !it->pending && it->id != id_keep && it->t_used < (it->root ? cut_rt : cut)) {
            n_removed++;
            n_roots += it->root ? 1 : 0;
            n_bytes += it->size;
            erase_entry(it);
        }
        it = next;
    }

    if (n_removed > 0) {
        SRV_INF("disk prompt cache: removed %zu expired entries (%zu root) unused for more than %.1f h (roots: %.1f h), %.3f MiB\n",
                n_removed, n_roots, ttl_hours, root_ttl_hours, n_bytes / MiB);
    }
}

bool server_prompt_cache_disk::make_room(size_t size_new, int64_t id_keep, time_point t_new) {
    reap_expired(id_keep);

    if (limit_size > 0 && size_new > limit_size) {
        return false;
    }

    // bytes to free for the size limit, and to keep the reserve free on the file system
    size_t need = 0;

    const size_t size_cur = size_on_disk();
    if (limit_size > 0 && size_cur + size_new > limit_size) {
        need = size_cur + size_new - limit_size;
    }

    if (size_new > 0) {
        std::error_code ec;
        const auto space = fs::space(fs::u8path(dir), ec);
        const size_t reserve = ec ? 0 : disk_reserve(space);
        if (!ec && space.available < size_new + reserve) {
            need = std::max<size_t>(need, size_new + reserve - space.available);
        }
    }

    if (need == 0) {
        return true;
    }

    // conversation entries and files of other fingerprints before root entries, least recently used first within each,
    // and only files used before the new state
    std::vector<std::list<entry>::iterator> lru;
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->on_disk && !it->busy() && !it->pending && it->id != id_keep && it->t_used < t_new) {
            lru.push_back(it);
        }
    }
    // root entries are kept last while they hold at most a quarter of the limit; the oldest roots beyond that budget
    // compete with the other entries (one root per distinct system prompt, so they could otherwise fill the cache)
    std::vector<std::list<entry>::iterator> roots;
    size_t root_bytes = 0;
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->on_disk && it->root) {
            roots.push_back(it);
            root_bytes += it->size;
        }
    }
    std::sort(roots.begin(), roots.end(), [](const auto & a, const auto & b) { return a->t_used < b->t_used; });
    std::vector<const entry *> demoted;
    const size_t root_budget = limit_size > 0 ? limit_size / 4 : SIZE_MAX;
    for (size_t i = 0; i < roots.size() && root_bytes > root_budget; ++i) {
        demoted.push_back(&*roots[i]);
        root_bytes -= roots[i]->size;
    }
    const auto protected_root = [&demoted](const entry & e) {
        return e.root && std::find(demoted.begin(), demoted.end(), &e) == demoted.end();
    };
    std::sort(lru.begin(), lru.end(), [&protected_root](const auto & a, const auto & b) {
        const bool pa = protected_root(*a), pb = protected_root(*b);
        if (pa != pb) {
            return !pa; // protected root entries last
        }
        return a->t_used < b->t_used;
    });

    // count the freed bytes from the file sizes: a copy-on-write file system can report the free space later
    size_t n_free  = 0;
    size_t n_evict = 0;
    while (n_evict < lru.size() && n_free < need) {
        n_free += lru[n_evict++]->size;
    }

    if (n_free < need) {
        return false;
    }

    for (size_t i = 0; i < n_evict; ++i) {
        SRV_INF("disk prompt cache: removing least recently used %s with %zu tokens (%.3f MiB)\n",
                lru[i]->foreign ? "file of another fingerprint" : (lru[i]->root ? "root prompt" : "prompt"), lru[i]->tokens.size(), lru[i]->size / MiB);

        erase_entry(lru[i]);
    }

    return true;
}

void server_prompt_cache_disk::scan() {
    std::error_code ec;
    fs::create_directories(fs::u8path(dir), ec);
    if (ec) {
        SRV_WRN("disk prompt cache: cannot create '%s': %s\n", dir.c_str(), ec.message().c_str());
        return;
    }

    try {
        for (const auto & de : fs::directory_iterator(fs::u8path(dir))) {
            const std::string name = fs_path_to_utf8(de.path().filename());
            const std::string path = fs_path_to_utf8(de.path());

            if (name.rfind("pc-", 0) != 0 || !de.is_regular_file()) {
                continue;
            }

            const bool own = name.rfind(prefix, 0) == 0;

            if (ends_with(name, ".tmp")) {
                // left behind by an interrupted write (another fingerprint may still be writing its own)
                if (own) {
                    fs::remove(de.path(), ec);
                }
                continue;
            }

            if (!ends_with(name, ".bin")) {
                continue;
            }

            if (!own) {
                entries.emplace_back();

                auto & e = entries.back();
                e.id      = id_next++;
                e.path    = path;
                e.size    = de.file_size();
                e.t_used  = de.last_write_time();
                e.on_disk = true;
                e.foreign = true;

                continue;
            }

            server_prompt_cache_state state;
            if (!read_file(path, state, true)) {
                SRV_WRN("disk prompt cache: removing unreadable file '%s'\n", path.c_str());
                fs::remove(de.path(), ec);
                continue;
            }

            entries.emplace_back();

            auto & e = entries.back();
            e.id      = id_next++;
            e.path    = path;
            e.tokens  = std::move(state.prompt.tokens);
            e.ckpts   = ckpt_positions(state.prompt);
            e.root    = state.root;
            e.size    = de.file_size();
            e.t_used  = de.last_write_time();
            e.on_disk = true;
        }
    } catch (const std::exception & err) {
        SRV_WRN("disk prompt cache: failed to scan '%s': %s\n", dir.c_str(), err.what());
    }

    make_room(0, -1, time_point::max());

    const size_t n_own = std::count_if(entries.begin(), entries.end(), [](const entry & e) { return !e.foreign; });

    SRV_INF("disk prompt cache: %zu prompt(s), %zu file(s) of other fingerprints, %.3f MiB in '%s' (limit: %.3f MiB)\n",
            n_own, entries.size() - n_own, size_on_disk() / MiB, dir.c_str(), limit_size / MiB);
}

void server_prompt_cache_disk::put(server_prompt_cache_state && state, time_point t_used, bool bounded) {
    const size_t n_tokens = state.prompt.tokens.size();
    if (n_tokens == 0) {
        return;
    }

    std::vector<char> packed;
    try {
        packed = state.prompt.tokens.serialize();
    } catch (const std::exception & err) {
        SRV_WRN("disk prompt cache: cannot store prompt with %zu tokens: %s\n", n_tokens, err.what());
        return;
    }

    const std::string name = prefix + hex64(fnv1a(FNV_OFFSET, packed.data(), packed.size())) + ".bin";

    const std::vector<int64_t> ckpts = ckpt_positions(state.prompt);

    std::lock_guard<std::mutex> lk(mtx);

    // a root state is stored as its own (small) file even when a conversation entry contains it: conversation
    // entries expire (LLAMA_CACHE_DISK_TTL) and the root must outlive them
    for (const auto & e : entries) {
        if (!e.foreign && (!state.root || e.root) && covers(e.tokens, e.ckpts, state.prompt.tokens, ckpts)) {
            SRV_TRC("disk prompt cache: prompt with %zu tokens is already stored, skipping\n", n_tokens);
            return;
        }
    }

    // waiting states that this one continues would be removed once it is written, drop them now
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);
        if (!it->on_disk && !it->busy() && (!it->root || state.root) && covers(state.prompt.tokens, ckpts, it->tokens, it->ckpts)) {
            erase_entry(it);
        }
        it = next;
    }

    const size_t size_state = state.size();

    // the disk is behind: drop the oldest waiting states to bound the memory they hold
    while (bounded && limit_pending > 0 && size_pending + size_state > limit_pending) {
        auto it_old = entries.end();
        for (const int64_t id : queue) {
            auto it = find_entry(id);
            if (it != entries.end() && !it->busy()) {
                it_old = it;
                break;
            }
        }

        if (it_old == entries.end()) {
            break;
        }

        SRV_WRN("disk prompt cache: writes are behind, dropping waiting prompt with %zu tokens (%.3f MiB)\n",
                it_old->tokens.size(), it_old->size_mem / MiB);

        erase_entry(it_old);
    }

    entries.emplace_back();

    auto & e = entries.back();
    e.id       = id_next++;
    e.path     = fs_path_to_utf8(fs::u8path(dir) / fs::u8path(name));
    e.tokens   = state.prompt.tokens.clone();
    e.ckpts    = ckpts;
    e.root     = state.root;
    e.size     = size_state;
    e.size_mem = size_state;
    e.t_used   = t_used;
    e.pending  = std::make_unique<server_prompt_cache_state>(std::move(state));

    queue.push_back(e.id);
    size_pending += size_state;

    SRV_TRC("disk prompt cache: queued prompt with %zu tokens (%.3f MiB), %zu waiting (%.3f MiB)\n",
            n_tokens, size_state / MiB, queue.size(), size_pending / MiB);

    cv.notify_all();
}

bool server_prompt_cache_disk::contains(const server_prompt & prompt) const {
    const std::vector<int64_t> ckpts = ckpt_positions(prompt);

    std::lock_guard<std::mutex> lk(mtx);

    for (const auto & e : entries) {
        if (!e.foreign && covers(e.tokens, e.ckpts, prompt.tokens, ckpts)) {
            return true;
        }
    }

    return false;
}

size_t server_prompt_cache_disk::n_common_prefix(const server_tokens & tokens) const {
    std::lock_guard<std::mutex> lk(mtx);

    size_t res = 0;
    for (const auto & e : entries) {
        if (!e.foreign) {
            res = std::max(res, e.tokens.get_common_prefix(tokens));
        }
    }

    return res;
}

int64_t server_prompt_cache_disk::find(const server_tokens & tokens_new, size_t n_tokens_max, size_t & lcp_best, bool rolls_back) const {
    std::lock_guard<std::mutex> lk(mtx);

    int64_t id_best = -1;

    for (const auto & e : entries) {
        if (e.foreign || e.n_fail >= 2 || e.tokens.size() > n_tokens_max) {
            continue;
        }

        size_t lcp_cur = e.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / e.tokens.size();

        if (rolls_back && lcp_cur < e.tokens.size()) {
            size_t usable = 0;
            for (const int64_t pos : e.ckpts) {
                if (pos > 0 && (size_t) pos <= lcp_cur) {
                    usable = std::max(usable, (size_t) pos);
                }
            }
            lcp_cur = usable;
        }

        SRV_TRC("   - disk prompt with length %7zu, restores %7zu, f_keep = %.3f\n", e.tokens.size(), lcp_cur, f_keep_cur);

        // same rule as the RAM level; a memory that rolls back to checkpoints restores a copy (the file stays), so
        // any part of a long prompt is worth it
        if (!rolls_back && f_keep_cur < 0.25f) {
            continue;
        }

        if (lcp_cur > lcp_best) {
            lcp_best = lcp_cur;
            id_best  = e.id;
        }
    }

    return id_best;
}

bool server_prompt_cache_disk::fetch(int64_t id, server_prompt_cache_state & out) {
    std::unique_lock<std::mutex> lk(mtx);

    auto it = find_entry(id);
    if (it == entries.end()) {
        return false;
    }

    if (it->writing) {
        // stop the write at the next chunk, the writer keeps the state for us
        it->cancel = true;
        cv.wait(lk, [&]() {
            auto cur = find_entry(id);
            return cur == entries.end() || !cur->writing;
        });

        it = find_entry(id);
        if (it == entries.end()) {
            return false;
        }
    }

    if (it->pending) {
        SRV_INF("disk prompt cache: taking back prompt with %zu tokens before it is written\n", it->tokens.size());

        out = std::move(*it->pending);

        it->pending.reset();
        size_pending -= it->size_mem;
        it->cancel = false;

        queue.erase(std::remove(queue.begin(), queue.end(), id), queue.end());

        if (!it->on_disk) {
            entries.erase(it);
        }

        cv.notify_all();

        return true;
    }

    it->reading = true;

    const std::string path = it->path;

    lk.unlock();

    const int64_t t_start = ggml_time_us();

    // the entry is not erased while it is being read, so it stays valid
    bool ok      = false;
    bool corrupt = false;
    try {
        ok      = read_file(path, out, false);
        corrupt = !ok;
    } catch (const std::exception & err) {
        SRV_WRN("disk prompt cache: failed to read '%s': %s\n", path.c_str(), err.what());
    }

    lk.lock();

    it->reading = false;

    if (!ok) {
        if (corrupt) {
            SRV_WRN("disk prompt cache: '%s' is corrupt, removing it\n", path.c_str());
            erase_entry(it);
        }
        cv.notify_all();
        return false;
    }

    it->t_used = time_point::clock::now();

    std::error_code ec;
    fs::last_write_time(fs::u8path(path), it->t_used, ec);

    SRV_INF("disk prompt cache: read prompt with %zu tokens (%.3f MiB) in %.2f s\n",
            it->tokens.size(), it->size / MiB, (ggml_time_us() - t_start) / 1e6);

    cv.notify_all();

    return true;
}

void server_prompt_cache_disk::restore_failed(int64_t id, server_prompt_cache_state && state) {
    {
        std::lock_guard<std::mutex> lk(mtx);

        auto it = find_entry(id);
        if (it != entries.end() && it->on_disk) {
            it->n_fail++;

            SRV_WRN("disk prompt cache: prompt with %zu tokens did not restore (%d time(s))%s\n",
                    it->tokens.size(), it->n_fail, it->n_fail >= 2 ? ", not using it any more" : "");

            return;
        }
    }

    // it was taken back from the queue, write it after all
    put(std::move(state));
}

void server_prompt_cache_disk::flush() {
    std::unique_lock<std::mutex> lk(mtx);

    cv.wait(lk, [this]() {
        return queue.empty() && std::none_of(entries.begin(), entries.end(), [](const entry & e) { return e.writing; });
    });
}

void server_prompt_cache_disk::worker_loop() {
    std::unique_lock<std::mutex> lk(mtx);

    while (true) {
        cv.wait(lk, [this]() { return stopping || !queue.empty(); });

        if (queue.empty()) {
            // stopping, and nothing left to write
            break;
        }

        const int64_t id = queue.front();
        queue.pop_front();

        auto it = find_entry(id);
        if (it == entries.end() || !it->pending) {
            continue;
        }

        if (!make_room(it->size_mem, id, it->t_used)) {
            SRV_WRN("disk prompt cache: no room for prompt with %zu tokens (%.3f MiB), dropping it\n", it->tokens.size(), it->size_mem / MiB);
            erase_entry(it);
            cv.notify_all();
            continue;
        }

        it->writing = true;

        const std::string path     = it->path;
        const std::string path_tmp = path + ".tmp";

        const server_prompt_cache_state & state = *it->pending;

        const time_point t_used = it->t_used;

        lk.unlock();

        const int64_t t_start = ggml_time_us();

        bool ok = false;
        try {
            ok = write_file(path_tmp, state, it->cancel);
        } catch (const std::exception & err) {
            SRV_WRN("disk prompt cache: failed to write '%s': %s\n", path_tmp.c_str(), err.what());
        }

        std::error_code ec;
        if (ok) {
            fs::rename(fs::u8path(path_tmp), fs::u8path(path), ec);
            ok = !ec;
        }
        if (ok) {
            // the mtime keeps the LRU order over restarts
            fs::last_write_time(fs::u8path(path), t_used, ec);
        }
        if (!ok) {
            fs::remove(fs::u8path(path_tmp), ec);
        }

        const double t_s = (ggml_time_us() - t_start) / 1e6;

        lk.lock();

        it->writing = false;

        if (ok) {
            it->on_disk = true;
            it->size    = fs::file_size(fs::u8path(path), ec);

            SRV_INF("disk prompt cache: wrote prompt with %zu tokens (%.3f MiB) in %.2f s, %.3f MiB on disk\n",
                    it->tokens.size(), it->size / MiB, t_s, size_on_disk() / MiB);

            if (!it->cancel) {
                // keep the state if fetch() is waiting for it
                size_pending -= it->size_mem;
                it->pending.reset();
            }

            drop_superseded(*it);

            make_room(0, id, it->t_used);
        } else if (!it->cancel) {
            SRV_WRN("disk prompt cache: failed to write '%s', dropping prompt with %zu tokens\n", path.c_str(), it->tokens.size());
            erase_entry(it);
        }

        cv.notify_all();
    }
}

// file layout, native byte order:
//   u32 magic, u32 version, u64 fingerprint
//   u64 n, n x llama_token: server_tokens::serialize()
//   u8 root, u32 n_checkpoints, n_checkpoints x i64 n_tokens: read with the tokens to index the file
//   u64 n, n bytes: target state
//   u64 n, n bytes: draft state
//   u32 n_checkpoints, then per checkpoint:
//     i64 n_tokens, i32 pos_min, i32 pos_max, u8 flags (1 root, 2 shared), and the target, draft and speculative data as u64 n + n bytes

bool server_prompt_cache_disk::write_file(const std::string & path, const server_prompt_cache_state & state, const std::atomic<bool> & cancel) const {
    disk_file f(path, "wb");
    if (!f.ok) {
        return false;
    }

    const std::vector<char> packed = state.prompt.tokens.serialize();
    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);

    f.write_val<uint32_t>(DISK_MAGIC);
    f.write_val<uint32_t>(DISK_VERSION);
    f.write_val<uint64_t>(fingerprint);
    f.write_val<uint64_t>(packed.size() / sizeof(llama_token));
    f.write(packed.data(), packed.size());

    f.write_val<uint8_t>(state.root ? 1 : 0);
    f.write_val<uint32_t>((uint32_t) state.prompt.checkpoints.size());
    for (const auto & cur : state.prompt.checkpoints) {
        f.write_val<int64_t>(cur.n_tokens);
    }

    if (!write_blob(f, state.data.main, cancel) ||
        !write_blob(f, state.data.drft, cancel)) {
        return false;
    }

    f.write_val<uint32_t>((uint32_t) state.prompt.checkpoints.size());

    for (const auto & cur : state.prompt.checkpoints) {
        f.write_val<int64_t>  (cur.n_tokens);
        f.write_val<llama_pos>(cur.pos_min);
        f.write_val<llama_pos>(cur.pos_max);
        f.write_val<uint8_t>((cur.root ? 1 : 0) | (cur.shared ? 2 : 0));

        if (!write_blob(f, cur.data_tgt,  cancel) ||
            !write_blob(f, cur.data_dft,  cancel) ||
            !write_blob(f, cur.data_spec, cancel)) {
            return false;
        }
    }

    return f.sync();
}

bool server_prompt_cache_disk::read_file(const std::string & path, server_prompt_cache_state & out, bool tokens_only) const {
    std::error_code ec;
    const size_t file_size = fs::file_size(fs::u8path(path), ec);
    if (ec) {
        return false;
    }

    disk_file f(path, "rb");

    if (f.read_val<uint32_t>() != DISK_MAGIC   ||
        f.read_val<uint32_t>() != DISK_VERSION ||
        f.read_val<uint64_t>() != fingerprint  || !f.ok) {
        return false;
    }

    const uint64_t n_packed = f.read_val<uint64_t>();
    if (!f.ok || n_packed > (file_size - f.n) / sizeof(llama_token)) {
        return false;
    }

    llama_tokens packed(n_packed);
    f.read(packed.data(), n_packed * sizeof(llama_token));
    if (!f.ok) {
        return false;
    }

    try {
        out.prompt.tokens = server_tokens::deserialize(packed, has_mtmd);
    } catch (const std::exception & err) {
        SRV_WRN("disk prompt cache: invalid tokens in '%s': %s\n", path.c_str(), err.what());
        return false;
    }

    out.root = f.read_val<uint8_t>() != 0;

    const uint32_t n_index = f.read_val<uint32_t>();
    if (!f.ok || n_index > (file_size - f.n) / sizeof(int64_t)) {
        return false;
    }

    // the checkpoints with their positions only, for the index
    out.prompt.checkpoints.clear();
    for (uint32_t i = 0; i < n_index && f.ok; ++i) {
        common_prompt_checkpoint cur;
        cur.n_tokens = f.read_val<int64_t>();
        cur.pos_min  = 0;
        cur.pos_max  = 0;
        out.prompt.checkpoints.push_back(std::move(cur));
    }

    if (!f.ok) {
        return false;
    }

    if (tokens_only) {
        return true;
    }

    if (!read_blob(f, out.data.main, file_size) ||
        !read_blob(f, out.data.drft, file_size)) {
        return false;
    }

    const uint32_t n_checkpoints = f.read_val<uint32_t>();
    if (n_checkpoints != n_index) {
        return false;
    }

    out.prompt.checkpoints.clear();

    for (uint32_t i = 0; i < n_checkpoints && f.ok; ++i) {
        common_prompt_checkpoint cur;

        cur.id_task  = -1; // not created by a task: a checkpoint that fails to load falls back to reprocessing
        cur.n_tokens = f.read_val<int64_t>();
        cur.pos_min  = f.read_val<llama_pos>();
        cur.pos_max  = f.read_val<llama_pos>();

        const uint8_t flags = f.read_val<uint8_t>();
        cur.root   = flags & 1;
        cur.shared = flags & 2;

        if (!read_blob(f, cur.data_tgt,  file_size) ||
            !read_blob(f, cur.data_dft,  file_size) ||
            !read_blob(f, cur.data_spec, file_size) || cur.data_tgt.empty()) {
            return false;
        }

        out.prompt.checkpoints.push_back(std::move(cur));
    }

    return f.ok && f.n == file_size;
}

static bool file_fingerprint(uint64_t & h, const std::string & path) {
    std::error_code ec;
    const uint64_t size = fs::file_size(fs::u8path(path), ec);
    if (ec) {
        return false;
    }

    disk_file f(path, "rb");
    if (!f.ok) {
        return false;
    }

    h = fnv1a(h, &size, sizeof(size));

    // 16 samples of 4 KiB spread over the file tell apart models with the same size and metadata
    constexpr size_t n_samples   = 16;
    constexpr size_t sample_size = 4096;

    std::vector<uint8_t> buf(sample_size);

    for (size_t i = 0; i < n_samples; ++i) {
        const uint64_t offset = size > sample_size ? (size - sample_size) * i / (n_samples - 1) : 0;
        if (!f.seek(offset)) {
            return false;
        }

        const size_t n = fread(buf.data(), 1, sample_size, f.f);
        h = fnv1a(h, buf.data(), n);
    }

    return true;
}

uint64_t server_prompt_cache_disk::compute_fingerprint(const std::vector<std::string> & files, const std::string & config) {
    uint64_t h = fnv1a(FNV_OFFSET, &DISK_VERSION, sizeof(DISK_VERSION));

    for (const auto & path : files) {
        if (!file_fingerprint(h, path)) {
            SRV_WRN("disk prompt cache: cannot read '%s'\n", path.c_str());
            return 0;
        }
    }

    h = fnv1a(h, config.data(), config.size());

    return h;
}
