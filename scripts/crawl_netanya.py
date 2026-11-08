"""Polite, resumable crawler for https://www.netanya.muni.il/ that produces RAG-ready output.

Output (in --out, default rag-data/netanya):
  pages.jsonl      one JSON object per HTML page: url, title, description, breadcrumb, tags,
                   lang, text (clean Markdown of the main content), fetched_at
  documents.jsonl  one JSON object per linked PDF/DOCX/etc.: url, file, source_page, text
  docs/            the downloaded documents themselves
  raw/             gzipped raw HTML (so extraction can be re-run without re-crawling)
  state.json       crawl frontier, for resuming (just run the script again)

Usage:  python scripts/crawl_netanya.py [--out DIR] [--max-pages N] [--delay SEC]
"""
import argparse, gzip, hashlib, io, json, os, re, sys, time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urlunparse, unquote, quote, parse_qsl

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

HOSTS = {"www.netanya.muni.il", "netanya.muni.il"}
START = ["https://www.netanya.muni.il/", "https://www.netanya.muni.il/Pages/allServices.aspx",
         "https://www.netanya.muni.il/news/Pages/Allreports.aspx"]
DOC_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".rtf", ".txt", ".csv"}
SKIP_EXT = {".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
            ".mp4", ".mp3", ".wav", ".avi", ".mov", ".zip", ".rar", ".7z", ".woff", ".woff2",
            ".ttf", ".eot", ".xml", ".axd", ".ashx", ".json", ".dwg", ".exe", ".msi"}
SKIP_PATH = re.compile(r"/(_layouts|_vti_bin|_catalogs|_api|style%20library|style library|"
                       r"siteassets/js|_controltemplates|lists/[^/]+/(allitems|dispform|editform|newform))",
                       re.I)
SKIP_PAGE = re.compile(r"(search\.aspx|osssearchresults|login|signin|authenticate|print)", re.I)
MAX_QUERY_VARIANTS = 15        # per path, to avoid infinite filter/calendar URLs
MAX_DOC_BYTES = 30 * 1024 * 1024
UA = "KortexRAGCrawler/1.0 (+contact: site owner; polite crawl for municipal info assistant)"


def norm(url):
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return None
    host = p.netloc.lower().split(":")[0]
    if host not in HOSTS:
        return None
    path = quote(unquote(p.path), safe="/%:@&=+$,;~-._!*'()")
    q = "&".join(f"{k}={v}" for k, v in sorted(parse_qsl(p.query, keep_blank_values=True))
                 if not k.lower().startswith(("utm_", "source", "fbclid")))
    return urlunparse(("https", "www.netanya.muni.il", path or "/", "", q, ""))


def ext_of(url):
    return os.path.splitext(unquote(urlparse(url).path).lower())[1]


# Israeli ID numbers written next to an ID label ("ת.ז 312480049", "מספר זהות: 0123..."). Only numbers
# with a label are removed, so budget codes, permit numbers and company numbers (ח.פ) stay intact.
ID_WITH_LABEL = re.compile(r"((?:ת\.ז\.?|ת\"ז|ת''ז|תעודת זהות|מספר זהות|מס' זהות|מס\. זהות)\s*[:.\-]?\s*)\d{8,9}(?!\d)")
# Empty contact-card template: "## פרטי התקשרות / * טלפון / * פקס / ... / * שעות קבלה" with no values
# (label lines only: short, no digits). Real contact details contain digits, so they never match.
CONTACT_TEMPLATE = re.compile(r"## פרטי התקשרות\n(?:[^\d\n]{0,90}\n)*?\* שעות קבלה\n?")
CONTACT_FORM = re.compile(r"## יצירת קשר\n+\\\*כוכבית הינה שדה חובה[\s\S]*?(?:\[לאנשי קשר\]\([^)]*\)|$)")


def redact(text):
    return ID_WITH_LABEL.sub(r"\1[הוסר]", text)


def strip_boilerplate(md):
    """Drop the site's empty contact-card template and the contact form (labels only, no data)."""
    md = CONTACT_TEMPLATE.sub("", md)
    return CONTACT_FORM.sub("", md)


def clean_text(md):
    md = redact(strip_boilerplate(md))
    md = re.sub(r"[​‎‏﻿]", "", md)
    md = re.sub(r"<img[^>]*>", "", md)
    md = re.sub(r"\[\s*\]\([^)]*\)", "", md)               # empty links
    md = re.sub(r"(?m)^\s*xml version=.*$", "", md)
    md = re.sub(r"(?m)^#{1,6}\s*$", "", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


SP_TAXONOMY = re.compile(r"L\d\|#0?[0-9a-fA-F-]{36}\|([^;|]+)")
SP_JUNK = [re.compile(p) for p in (
    r"\b(?:GTSet|GPP|GP0|L\d)\|#0?[0-9a-fA-F-]{36};?",      # taxonomy GUID tokens
    r"\b(?:string|float|int|datetime|boolean|integer|number);#",  # SharePoint field-type prefixes
    r"^\d+;#",                                              # lookup ids ("12;#Value")
)]


def clean_sp(s):
    """Turn SharePoint list-field values into plain text (keeps taxonomy labels, drops GUIDs)."""
    if "#" not in s:
        return s
    labels = SP_TAXONOMY.findall(s)
    if labels:
        return "; ".join(l.strip() for l in labels)
    for p in SP_JUNK:
        s = p.sub("", s)
    return s.replace(";#", "; ").strip()


TEXT_URL = re.compile(r"""(?:https?://(?:www\.)?netanya\.muni\.il)?/[^\s"'<>|]+?\.(?:pdf|docx?|xlsx?|pptx?|rtf)\b""", re.I)

NOISE_TEXT = ("Getting your", "Trinity Audio", "player ready", "JavaScript אינו זמין",
              "חזרה לראש הדף", "סגור חלון")


def extract_page(html, url):
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)
    descs = [m.get("content", "").strip() for m in soup.find_all("meta", attrs={"name": "description"})]
    description = " | ".join(d for d in dict.fromkeys(descs) if d)
    bc = soup.select_one("[class*=readcrumb]")
    breadcrumb = [x.strip() for x in bc.get_text(" > ", strip=True).split(">") if x.strip()] if bc else []
    tags_el = soup.find(string=re.compile("תגיות לחיפוש"))
    tags = []
    if tags_el and tags_el.parent:
        box = tags_el.parent.parent or tags_el.parent
        tags = [a.get_text(strip=True) for a in box.find_all("a") if a.get_text(strip=True)]
    lang = (soup.html.get("lang") if soup.html else "") or ""

    links = [a.get("href") for a in soup.find_all("a", href=True)]
    links += [f.get("src") for f in soup.find_all(["iframe"], src=True)]
    # document URLs that appear only as plain text (e.g. SharePoint list columns)
    links += [m.group(0) for m in TEXT_URL.finditer(soup.get_text(" "))]

    main = (soup.select_one(".content_page_layout") or soup.select_one("#DeltaPlaceHolderMain")
            or soup.select_one("[role=main]") or soup.body or soup)
    for t in main.find_all(["script", "style", "noscript", "form", "input", "button", "select",
                            "textarea", "svg", "iframe", "nav", "header", "footer"]):
        t.decompose()
    for sel in ["[class*=readcrumb]", "[class*=modal]", "[class*=popup]", "[class*=share]",
                "[class*=language]", "[class*=lang_]", "[class*=print]", "[class*=trinity]",
                "[style*='display:none']", "[style*='display: none']", "[aria-hidden=true]"]:
        for t in main.select(sel):
            t.decompose()
    for t in main.find_all(string=re.compile(r"\{\{.*?\}\}")):
        par = t.parent
        if par is not None:
            par.decompose()
    for t in list(main.find_all(string=re.compile(r"[;|]#"))):
        t.replace_with(clean_sp(str(t)))
    md = markdownify(str(main), heading_style="ATX", strip=["img"])
    lines = [l for l in md.splitlines() if not any(n in l for n in NOISE_TEXT)]
    text = clean_text("\n".join(lines))
    return {"url": url, "title": title, "description": description, "breadcrumb": breadcrumb,
            "tags": tags, "lang": lang, "text": text}, links


def extract_doc(path, ext):
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            r = PdfReader(path)
            return clean_text("\n\n".join((p.extract_text() or "") for p in r.pages))
        if ext == ".docx":
            import docx
            d = docx.Document(path)
            parts = [p.text for p in d.paragraphs]
            for tbl in d.tables:
                for row in tbl.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
            return clean_text("\n".join(parts))
        if ext in (".xlsx", ".xls"):
            rows_out = []
            if ext == ".xlsx":
                import openpyxl
                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                sheets = [(ws.title, ws.iter_rows(values_only=True)) for ws in wb.worksheets]
            else:
                import xlrd
                wb = xlrd.open_workbook(path)
                sheets = [(sh.name, (sh.row_values(i) for i in range(sh.nrows))) for sh in wb.sheets()]
            for name, rows in sheets:
                rows_out.append(f"## {name}")
                for row in rows:
                    cells = ["" if c is None else clean_sp(str(c).strip()) for c in row]
                    if any(cells):
                        rows_out.append(" | ".join(cells).rstrip(" |"))
            return clean_text("\n".join(rows_out))
        if ext == ".pptx":
            from pptx import Presentation
            parts = []
            for i, slide in enumerate(Presentation(path).slides, 1):
                parts.append(f"## Slide {i}")
                parts += [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame]
            return clean_text("\n".join(parts))
        if ext == ".doc":
            # Legacy Word: no pure-Python parser; pull UTF-16LE text runs out of the binary.
            data = open(path, "rb").read()
            runs = re.findall(rb"(?:[\x20-\x7e\r\n\t]\x00|[\xd0-\xea]\x05){4,}", data)
            return clean_text("\n".join(r.decode("utf-16-le", "ignore") for r in runs).replace("\r", "\n"))
        if ext in (".txt", ".csv"):
            return clean_text(open(path, encoding="utf-8", errors="replace").read())
    except Exception as e:  # corrupt/encrypted files
        return f"[extraction failed: {e}]"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("rag-data", "netanya"))
    ap.add_argument("--max-pages", type=int, default=20000)
    ap.add_argument("--delay", type=float, default=0.4)
    ap.add_argument("--no-docs", action="store_true")
    ap.add_argument("--reprocess", action="store_true",
                    help="re-extract pages.jsonl from raw/ with the current cleaning rules and "
                         "queue any newly found links, then continue crawling")
    ap.add_argument("--parse-docs", action="store_true",
                    help="extract text from the downloaded files in docs/ into documents.jsonl "
                         "(not done during the crawl); use with --max-pages 1 to only parse")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(os.path.join(a.out, "docs"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    state_f = os.path.join(a.out, "state.json")
    if os.path.exists(state_f):
        st = json.load(open(state_f, encoding="utf-8"))
        seen, queue, qvar = set(st["seen"]), deque(st["queue"]), st["qvar"]
        n_pages, n_docs = st["n_pages"], st["n_docs"]
        print(f"resuming: {n_pages} pages, {n_docs} docs, {len(queue)} queued")
    else:
        seen, queue, qvar, n_pages, n_docs = set(), deque(), {}, 0, 0
        for s in START:
            u = norm(s); seen.add(u); queue.append([u, None])
    pages_path = os.path.join(a.out, "pages.jsonl")
    docs_path = os.path.join(a.out, "documents.jsonl")
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept-Language": "he,en;q=0.8"})
    content_hashes = set()

    # What is already on disk wins over state.json (which can lag up to 30 s behind a kill).
    def load_jsonl(p):
        if not os.path.exists(p):
            return []
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    pages = {r["url"]: r for r in load_jsonl(pages_path)}          # dedupes by URL, keeps last
    docs = {r["url"]: r for r in load_jsonl(docs_path)}
    found_links = []
    if a.reprocess:
        for u, r in list(pages.items()):
            raw = os.path.join(a.out, "raw", hashlib.sha1(u.encode()).hexdigest() + ".html.gz")
            if not os.path.exists(raw):
                continue
            with gzip.open(raw, "rt", encoding="utf-8") as g:
                rec, links = extract_page(g.read(), u)
            rec["fetched_at"] = r.get("fetched_at")
            pages[u] = rec
            found_links += [(l, u) for l in links]
    if a.parse_docs:                                 # separate step: extract text from docs/
        for i, d in enumerate(docs.values(), 1):
            fp = os.path.join(a.out, d["file"])
            if os.path.exists(fp):
                d["text"] = extract_doc(fp, os.path.splitext(fp)[1].lower())
            if i % 50 == 0:
                print(f"parsed {i}/{len(docs)} documents", flush=True)
    kept = {}
    for u, r in pages.items():                                       # drop duplicate content
        h = hashlib.sha1(r["text"].encode()).hexdigest()
        if h not in content_hashes and len(r["text"]) >= 40:
            content_hashes.add(h); kept[u] = r
    for p, recs in ((pages_path, kept.values()), (docs_path, docs.values())):
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
    done = set(pages) | set(docs)
    queue = deque(item for item in queue
                  if item[0] not in done and "{{" not in unquote(item[0]))
    # retry URLs that failed on connection errors/timeouts (not 404s) in an earlier run
    err_path = os.path.join(a.out, "errors.log")
    if os.path.exists(err_path):
        queued = {item[0] for item in queue}
        for line in open(err_path, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if f[0] == "ERR" and f[1] not in done and f[1] not in queued:
                queue.append([f[1], None]); queued.add(f[1])
    seen |= done
    n_pages, n_docs = len(kept), len(docs)
    print(f"on disk: {n_pages} pages, {n_docs} docs; {len(queue)} queued", flush=True)

    pages_f = open(pages_path, "a", encoding="utf-8")
    docs_f = open(docs_path, "a", encoding="utf-8")
    err_f = open(os.path.join(a.out, "errors.log"), "a", encoding="utf-8")

    def save_state():
        json.dump({"seen": list(seen), "queue": list(queue), "qvar": qvar,
                   "n_pages": n_pages, "n_docs": n_docs}, open(state_f + ".tmp", "w", encoding="utf-8"))
        os.replace(state_f + ".tmp", state_f)

    def enqueue(href, base):
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "whatsapp:")):
            return
        if "{{" in href or "%7B%7B" in href.upper():   # unrendered Angular template placeholders
            return
        u = norm(urljoin(base, href.strip()))
        if not u or u in seen:
            return
        p = urlparse(u)
        e = ext_of(u)
        if e in SKIP_EXT or SKIP_PATH.search(unquote(p.path)) or SKIP_PAGE.search(p.path):
            return
        if e in DOC_EXT and a.no_docs:
            return
        if p.query:
            k = p.path.lower()
            if qvar.get(k, 0) >= MAX_QUERY_VARIANTS:
                return
            qvar[k] = qvar.get(k, 0) + 1
        seen.add(u)
        queue.append([u, base])

    before = len(queue)
    for l, b in found_links:
        enqueue(l, b)
    if found_links:
        print(f"reprocess: {len(queue) - before} new links queued from saved pages", flush=True)

    last_save = time.time()
    try:
        while queue and n_pages < a.max_pages:
            url, src = queue.popleft()
            time.sleep(a.delay)
            try:
                r = sess.get(url, timeout=40, stream=True)
                ctype = r.headers.get("Content-Type", "").lower()
                if r.status_code != 200:
                    err_f.write(f"{r.status_code}\t{url}\t{src}\n"); r.close(); continue
                final = norm(r.url) or url
                e = ext_of(final)
                is_doc = e in DOC_EXT or "pdf" in ctype or "officedocument" in ctype or "msword" in ctype
                if is_doc:
                    if a.no_docs:
                        r.close(); continue
                    data = b""
                    for chunk in r.iter_content(65536):
                        data += chunk
                        if len(data) > MAX_DOC_BYTES:
                            break
                    if len(data) > MAX_DOC_BYTES:
                        err_f.write(f"too-big\t{url}\n"); continue
                    if not e or e not in DOC_EXT:
                        e = ".pdf" if "pdf" in ctype else ".docx" if "wordprocessing" in ctype else ".bin"
                    base = re.sub(r"[^\w\-.]+", "_", os.path.basename(unquote(urlparse(final).path)))[-80:]
                    fname = hashlib.sha1(final.encode()).hexdigest()[:10] + "_" + (base or "file" + e)
                    fpath = os.path.join(a.out, "docs", fname)
                    open(fpath, "wb").write(data)
                    docs_f.write(json.dumps({"url": final, "file": "docs/" + fname, "source_page": src,
                                             "bytes": len(data), "content_type": ctype,
                                             "ext": e, "text": None,
                                             "fetched_at": datetime.now(timezone.utc).isoformat()},
                                            ensure_ascii=False) + "\n")
                    n_docs += 1
                    continue
                if "html" not in ctype:
                    r.close(); continue
                r.encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "utf-8"
                html = r.text
                rec, links = extract_page(html, final)
                for l in links:
                    enqueue(l, final)
                h = hashlib.sha1(rec["text"].encode()).hexdigest()
                if h in content_hashes or len(rec["text"]) < 40:
                    continue  # duplicate (same page under another URL) or empty listing shell
                content_hashes.add(h)
                with gzip.open(os.path.join(a.out, "raw", hashlib.sha1(final.encode()).hexdigest() + ".html.gz"), "wt", encoding="utf-8") as g:
                    g.write(html)
                rec["fetched_at"] = datetime.now(timezone.utc).isoformat()
                pages_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pages += 1
                if n_pages % 25 == 0:
                    pages_f.flush(); docs_f.flush(); err_f.flush()
                    print(f"[{datetime.now():%H:%M:%S}] pages={n_pages} docs={n_docs} queued={len(queue)} last={unquote(final)[:100]}", flush=True)
            except Exception as ex:
                err_f.write(f"ERR\t{url}\t{type(ex).__name__}: {ex}\n")
            if time.time() - last_save > 30:
                save_state(); last_save = time.time()
    finally:
        save_state(); pages_f.close(); docs_f.close(); err_f.close()
        print(f"done: pages={n_pages} docs={n_docs} still queued={len(queue)}")


if __name__ == "__main__":
    main()
