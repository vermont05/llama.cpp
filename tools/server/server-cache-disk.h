#pragma once

#include "server-task.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// disk level of the prompt cache (--cache-disk)
//
// prompt states evicted from the RAM level are written to one file each in the cache directory and are
// loaded back when a later request shares their prefix. the files are indexed again on startup, so the
// cache outlives the process. files are written by a background thread: an evicted state stays in memory
// until its file is complete, and a request that needs it in the meantime takes it back from the queue.
//
// file names start with a fingerprint of the model files and of the settings that change the KV contents,
// so several models can share one directory. files of other fingerprints are never loaded, but they count
// toward the size limit and are removed first when they are the least recently used.
struct server_prompt_cache_disk {
    using time_point = std::filesystem::file_time_type;

    // limit_size:    bytes in the directory, 0 = no limit
    // limit_pending: bytes of states waiting to be written, 0 = no limit; beyond it the oldest waiting state is dropped
    server_prompt_cache_disk(const std::string & dir, size_t limit_size, size_t limit_pending, uint64_t fingerprint, bool has_mtmd);

    // writes the states still waiting before returning
    ~server_prompt_cache_disk();

    // take a state evicted from RAM and write it in the background, t_used is its last use for the LRU order
    // nothing is written if an entry already holds these tokens or a continuation of them
    // bounded = false skips the limit on waiting states (the caller only moves states that are already in memory)
    void put(server_prompt_cache_state && state, time_point t_used = time_point::clock::now(), bool bounded = true);

    // true if an entry (written or waiting) holds these tokens or a continuation of them, with all their checkpoints
    bool contains(const server_prompt & prompt) const;

    // id of the entry that restores a longer prefix than lcp_best of tokens_new, -1 if none
    // updates lcp_best; entries with more than n_tokens_max tokens are skipped
    // rolls_back: a memory that rolls back only to checkpoints, an entry restores up to its newest checkpoint within
    // the common prefix (all of it if tokens_new continues the entry), see server_prompt_cache::n_usable()
    int64_t find(const server_tokens & tokens_new, size_t n_tokens_max, size_t & lcp_best, bool rolls_back = false) const;

    // the longest common prefix of tokens with an entry, restorable or not
    size_t n_common_prefix(const server_tokens & tokens) const;

    // move the state of an entry into out: a waiting state is taken back, a written one is read from its file
    // a corrupt file is deleted
    bool fetch(int64_t id, server_prompt_cache_state & out);

    // the state fetched for id did not restore into the context (no room, ...): a file is kept but skipped
    // after repeated failures, a state that was taken back is queued again
    void restore_failed(int64_t id, server_prompt_cache_state && state);

    // block until all waiting states are written
    void flush();

    size_t n_entries() const;

    // bytes of the files in the directory
    size_t size() const;

    const std::string & get_dir() const { return dir; }
    uint64_t get_fingerprint()    const { return fingerprint; }

    // hash of the size and sampled content of the files and of the config string, 0 if a file cannot be read
    static uint64_t compute_fingerprint(const std::vector<std::string> & files, const std::string & config);

private:
    struct entry {
        int64_t     id;
        std::string path;

        server_tokens tokens;

        // positions (n_tokens) of the checkpoints of the state
        std::vector<int64_t> ckpts;

        size_t size     = 0; // file size once written, state size while waiting
        size_t size_mem = 0; // bytes of the waiting state

        time_point t_used; // last use, kept as the file mtime

        bool on_disk = false;
        bool foreign = false; // file of another fingerprint: counts toward the limit, never loaded
        bool writing = false;
        bool reading = false;

        int n_fail = 0; // failed restores, the entry is skipped after 2

        // set by fetch() while the entry is being written, the writer then stops and keeps the state
        std::atomic<bool> cancel{false};

        // state waiting to be written
        std::unique_ptr<server_prompt_cache_state> pending;

        bool busy() const { return writing || reading; }
    };

    std::string dir;
    std::string prefix; // file name prefix for the current fingerprint

    size_t limit_size    = 0;
    size_t limit_pending = 0;

    uint64_t fingerprint = 0;
    bool     has_mtmd    = false;

    mutable std::mutex      mtx;
    std::condition_variable cv;

    std::list<entry>    entries;
    std::deque<int64_t> queue; // ids of the entries waiting to be written, oldest first

    size_t  size_pending = 0;
    int64_t id_next      = 0;
    bool    stopping     = false;

    std::thread worker;

    void worker_loop();

    std::list<entry>::iterator find_entry(int64_t id);

    void scan();

    // must be called with mtx held
    size_t size_on_disk() const;
    bool   make_room(size_t size_new, int64_t id_keep, time_point t_new);
    void   erase_entry(std::list<entry>::iterator it);
    void   drop_superseded(const entry & cur);

    // true if a holds the tokens of b or a continuation of them, and every checkpoint of b
    static bool covers(const server_tokens & a_tokens, const std::vector<int64_t> & a_ckpts,
                       const server_tokens & b_tokens, const std::vector<int64_t> & b_ckpts);

    bool write_file(const std::string & path, const server_prompt_cache_state & state, const std::atomic<bool> & cancel) const;
    bool read_file (const std::string & path, server_prompt_cache_state & out, bool tokens_only) const;
};
