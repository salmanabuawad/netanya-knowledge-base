"""Query the Netanya knowledge base built by scripts/build_netanya_kb.py (SQLite FTS5 / BM25).

Usage:  python scripts/search_netanya_kb.py "איך מגישים השגה על ארנונה" [-k 5] [--lang he]
            [--domain taxes_and_payments] [--no-archives] [--db PATH]
"""
import argparse, os, re, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_netanya_kb import DOMAIN_HE, HEB_PREFIX, NIQQUD, TIME_SENSITIVE  # noqa: E402

STOP = {"איך", "מה", "מתי", "איפה", "היכן", "למי", "של", "את", "על", "עם", "אני", "יש", "אם", "או",
        "גם", "כל", "זה", "זו", "לא", "האם", "אפשר", "ניתן", "רוצה", "צריך", "the", "a", "an", "of",
        "to", "in", "is", "how", "what", "where", "when", "do", "i", "for", "and", "or"}


def fts_query(q):
    terms = []
    for w in re.findall(r"\w+", NIQQUD.sub("", q.lower())):
        if w in STOP or len(w) < 2:
            continue
        alts = {w}
        m = HEB_PREFIX.match(w)
        if m and len(w) - m.end() >= 3:
            alts.add(w[m.end():])
        terms.append("(" + " OR ".join(f'"{t}"' for t in sorted(alts)) + ")")
    return " OR ".join(terms)


def search(db, q, k=5, lang=None, domain=None, no_archives=False):
    expr = fts_query(q)
    if not expr:
        return []
    sql = """SELECT c.id, c.title, c.heading, c.url, c.lang, c.kind, c.content,
                    bm25(chunks_fts, 4.0, 2.0, 1.0, 0.7) AS score
             FROM chunks_fts JOIN chunk_rowid r ON r.rowid = chunks_fts.rowid
             JOIN chunks c ON c.id = r.chunk_id
             WHERE chunks_fts MATCH ?"""
    params = [expr]
    if lang:
        sql += " AND c.lang = ?"
        params.append(lang)
    if domain:
        sql += " AND c.domain = ?"
        params.append(domain)
    if no_archives:
        sql += " AND c.category NOT IN (%s)" % ",".join("?" * len(TIME_SENSITIVE))
        params += TIME_SENSITIVE
    return db.execute(sql + " ORDER BY score LIMIT ?", params + [k]).fetchall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--lang")
    ap.add_argument("--domain", choices=sorted(DOMAIN_HE), help="limit to one municipal domain")
    ap.add_argument("--no-archives", action="store_true",
                    help="leave out news, events and tenders (time-sensitive content)")
    ap.add_argument("--db", default=os.path.join("rag-data", "netanya", "kb", "kb.sqlite"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    db = sqlite3.connect(a.db)
    for cid, title, heading, url, lang, kind, content, score in search(
            db, a.query, a.k, a.lang, a.domain, a.no_archives):
        print(f"[{score:.2f}] {title}" + (f" › {heading}" if heading else "") + f"  ({kind}, {lang})")
        print(f"  {url}")
        print("  " + content[:400].replace("\n", "\n  ") + ("…" if len(content) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
