"""Embed every knowledge-base chunk with a local Ollama embedding model (default bge-m3).

  python scripts/embed_netanya_kb.py [--kb rag-data/netanya/kb] [--model bge-m3] [--batch 16]

On CPU only, bge-m3 runs at ~0.4 chunks/s: ~6 h for the web pages and ~12 h more for documents.

Reads chunks from <kb>/kb.sqlite and stores unit-length float16 vectors in <kb>/vectors.sqlite,
keyed by chunk id + a hash of the embedded text. The vectors live in their own file so rebuilding
kb.sqlite (build_netanya_kb.py) keeps them: a re-run only embeds new or changed chunks and drops
vectors of chunks that no longer exist. Safe to interrupt and re-run.
Only a loopback Ollama URL is accepted, so chunk text never leaves this machine.
"""
import argparse, hashlib, json, os, sqlite3, sys, time, urllib.parse, urllib.request

import numpy as np

MAX_CHARS = 4000            # bge-m3 accepts 8k tokens; long chunks are rare (p95 ~1.6k chars)


def ollama_url(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != "http" or p.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise SystemExit(f"refusing non-local Ollama URL {url!r}: embeddings must stay on this machine")
    return url.rstrip("/")


def embed(texts, model, url, timeout=600):
    """Return unit-normalised float32 vectors for `texts` via Ollama /api/embed."""
    body = json.dumps({"model": model, "input": [t[:MAX_CHARS] for t in texts]}).encode()
    req = urllib.request.Request(url + "/api/embed", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        vecs = np.asarray(json.load(r)["embeddings"], dtype=np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)


def open_store(path, model):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS vectors(chunk_id TEXT PRIMARY KEY, text_sha1 TEXT, vec BLOB);
    """)
    old = dict(db.execute("SELECT key, value FROM meta"))
    if old.get("model") and old["model"] != model:
        raise SystemExit(f"{path} holds {old['model']} vectors; delete it to switch to {model}")
    db.execute("INSERT OR REPLACE INTO meta VALUES ('model', ?)", (model,))
    db.commit()
    return db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=os.path.join("rag-data", "netanya", "kb"))
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    url = ollama_url(a.url)

    kb = sqlite3.connect(f"file:{os.path.join(a.kb, 'kb.sqlite')}?mode=ro", uri=True)
    # Embedding is slow on CPU (~0.4 chunks/s with bge-m3), so do what residents ask about first:
    # service/info pages, then news/events/tenders, then documents (protocols, reports). Retrieval
    # uses whatever vectors exist and falls back to keyword search for the rest.
    chunks = {cid: text for cid, text in kb.execute("""
        SELECT id, text FROM chunks ORDER BY
          CASE WHEN kind = 'document' THEN 2
               WHEN category IN ('news_time_sensitive','events_time_sensitive','procurement_time_sensitive') THEN 1
               ELSE 0 END, source_id, ord""")}
    store = open_store(os.path.join(a.kb, "vectors.sqlite"), a.model)
    have = dict(store.execute("SELECT chunk_id, text_sha1 FROM vectors"))
    sha = {cid: hashlib.sha1(t.encode()).hexdigest() for cid, t in chunks.items()}
    gone = [c for c in have if c not in chunks]
    store.executemany("DELETE FROM vectors WHERE chunk_id = ?", [(c,) for c in gone])
    todo = [c for c in chunks if have.get(c) != sha[c]]
    print(f"{len(chunks):,} chunks: {len(chunks) - len(todo):,} already embedded, "
          f"{len(todo):,} to embed, {len(gone):,} stale vectors removed", flush=True)

    t0, done = time.time(), 0
    for i in range(0, len(todo), a.batch):
        ids = todo[i:i + a.batch]
        vecs = embed([chunks[c] for c in ids], a.model, url)
        store.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                          [(c, sha[c], v.astype(np.float16).tobytes()) for c, v in zip(ids, vecs)])
        store.commit()
        done += len(ids)
        if done % (a.batch * 20) == 0 or done == len(todo):
            rate = done / (time.time() - t0)
            print(f"  {done:,}/{len(todo):,}  {rate:.1f} chunks/s  ~{(len(todo) - done) / rate / 60:.0f} min left",
                  flush=True)
    dim = store.execute("SELECT length(vec) / 2 FROM vectors LIMIT 1").fetchone()
    if dim:
        store.execute("INSERT OR REPLACE INTO meta VALUES ('dim', ?)", (str(dim[0]),))
    store.commit()
    print(f"done: {store.execute('SELECT count(*) FROM vectors').fetchone()[0]:,} vectors ({a.model})")


if __name__ == "__main__":
    main()
