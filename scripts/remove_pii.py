"""Move downloaded documents whose main content is private individuals' data out of the RAG set.

  python -I scripts/remove_pii.py [--out rag-data/netanya] [--dry-run]

Moves matching files from <out>/docs/ to <out>-private/docs/ and their records from
documents.jsonl to <out>-private/documents.jsonl (with the reason). Nothing is deleted: to restore
a file, move it back and put its record back into documents.jsonl.

Documents that stay in the set but mention ID numbers (e.g. council protocols naming employees
"ת.ז 3124...") are handled by redact() in crawl_netanya.py when documents are parsed.
"""
import argparse, json, os, re, shutil, sys
from urllib.parse import unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crawl_netanya import extract_doc


def id_ok(s):
    t = 0
    for i, ch in enumerate(s.zfill(9)):
        v = int(ch) * (1 + i % 2); t += v - 9 if v > 9 else v
    return t % 10 == 0


def personal_ids(t):
    """9-digit numbers that pass the Israeli ID checksum, excluding company/nonprofit numbers
    (start with 5) and landline phone numbers."""
    return [m.group() for m in re.finditer(r"(?<!\d)\d{9}(?!\d)", t)
            if id_ok(m.group()) and m.group()[0] != "5" and not re.match(r"0[2-489]", m.group())]


def reason_for(name, text):
    head = text[:1500]
    if re.search(r"maakav", name, re.I):
        return "tree-permit register: private applicants' names and addresses"
    if re.search(r"העורר(?:ת|ים)?\s*:", head) or "ועדת הערר" in text[:400]:
        return "appeal decision naming private appellants and their property"
    if (re.search(r"(tocn|orat|oraot|orah)", name, re.I)
            and ("בעלי זכויות" in text or "בעלים" in text) and len(personal_ids(text)) >= 2):
        return "plan document listing landowners with ID numbers"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("rag-data", "netanya"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    priv = a.out.rstrip("/\\") + "-private"
    docs_path = os.path.join(a.out, "documents.jsonl")
    docs = [json.loads(l) for l in open(docs_path, encoding="utf-8") if l.strip()]
    keep, moved = [], []
    for d in docs:
        fp = os.path.join(a.out, d["file"])
        text = d.get("text")
        if not text and os.path.exists(fp):
            text = extract_doc(fp, os.path.splitext(fp)[1].lower()) or ""
        why = reason_for(unquote(os.path.basename(d["file"])), text or "")
        if why:
            moved.append({**d, "removed_reason": why})
            print(f"{'would move' if a.dry_run else 'moved'}: {unquote(d['file'])}  ({why})")
        else:
            keep.append(d)
    print(f"{len(moved)} of {len(docs)} documents {'flagged' if a.dry_run else 'moved'} to {priv}")
    if a.dry_run or not moved:
        return
    os.makedirs(os.path.join(priv, "docs"), exist_ok=True)
    with open(os.path.join(priv, "documents.jsonl"), "a", encoding="utf-8") as f:
        for d in moved:
            src = os.path.join(a.out, d["file"])
            if os.path.exists(src):
                shutil.move(src, os.path.join(priv, d["file"]))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(docs_path + ".tmp", "w", encoding="utf-8") as f:
        for d in keep:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(docs_path + ".tmp", docs_path)


if __name__ == "__main__":
    main()
