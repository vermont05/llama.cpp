IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Canonical working branch

`vermont-main` is the sole canonical working branch for PrismML / Swift Bonsai
work.  Begin all new work here and commit directly to it unless an experiment
needs a deliberately short-lived topic branch.

Current base: `6c5a46d` (`fix(qwen35): restore hadamard mtp embeddings`).  It
contains the previously named `swift-ptq1-prefill` work; that branch was
renamed rather than merged because it was already the checked-out canonical
history.

`/home/vermont/src/llama.cpp-prism-reference` is an audit-only reference
checkout.  Do not develop in it or merge it wholesale: it predates the Prism
optimization commits on `vermont-main`.  Copy a reference change only after a
focused comparison establishes that it is a missing compatibility fix.

Preserve the untracked `bin-cand-20261024/` binary snapshot.  It is not source
and must not be committed.
