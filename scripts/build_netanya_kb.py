"""Build a RAG knowledge base from the Netanya municipality crawl (scripts/crawl_netanya.py).

Input  (default rag-data/netanya):  pages.jsonl, documents.jsonl, docs/
Output (default rag-data/netanya/kb):
  doc_text.jsonl  cache of text extracted from docs/ (delete it to re-extract)
  sources.jsonl   one record per page/document: id, url, title, lang, section, kind, clean text
  chunks.jsonl    retrieval units (~1200 chars) with metadata; `text` is embedding-ready
                  (title + breadcrumb + heading prepended), `content` is the bare chunk
  kb.sqlite       sources + chunks tables and an FTS5 index (BM25) with Hebrew prefix folding
  REPORT.md       what is in the knowledge base and what could not be used
  ../netanya_knowledge_base.jsonl   the deliverable: every chunk in one file
Labelled Israeli ID numbers are removed with crawl_netanya.redact(); run scripts/remove_pii.py
first to move documents that are mainly private individuals' data out of the set.

Usage:  python scripts/build_netanya_kb.py [--data DIR] [--out DIR] [--target 1200]
Search: python scripts/search_netanya_kb.py "שעות פעילות מוקד 106"
"""
import argparse, collections, hashlib, json, os, re, sqlite3, sys, time
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crawl_netanya import redact  # noqa: E402  (removes labelled Israeli ID numbers)

# ---------------------------------------------------------------- page cleaning

# Sections that are site chrome, not content: related-news feeds, contact forms, filters.
DROP_SECTION = re.compile(
    r"^(חדשות בנושא|באותו נושא|קישורי סיכום|סינון לפי מתקנים|מפה ורשימה של שירותי|חפשו לפי שכונה|"
    r"חיפוש אירועים|אירועים פופולריים|כלל האירועים|יצירת קשר\s*$|Contact us|Relevant news|"
    r"Useful Links|Контакты|Статьи по данной теме|Nous contacter|Pour en savoir plus|Liens utiles)",
    re.I)
# Contact-widget headings: dropped, but what follows them (often a list of online forms) is kept.
DROP_HEADING = re.compile(r"^(פרטי התקשרות|ימי ושעות קבלת קהל)\s*$")
# Empty placeholders of the SharePoint contact widget (present on ~2,260 pages, never filled in).
PLACEHOLDER = {
    "* כתובת", "* ימי קבלה", "* שעות קבלה", "* טלפון", "* פקס", "* #### ימי ושעות פעילות:",
    "* + טלפון", "+ כתובת", "+ פקס", "+ ימי קבלה", "+ שעות קבלה", "+ חבר ב:", "מחלקות",
    "לא נמצאו מחלקות/אנשי קשר תחת מחלקה זו", "שינוי בטופס מרענן את החיפוש", "אגפים ויחידות",
    "חיפוש חופשי", "מועצה מינהלים וועדות", "טוען נתונים. אנא המתן...", "כן", "לא",
}
DROP_CELLS = {"תצוגת תאריך", "תמונת פריט", "True", "False", "Item Image", "Date Display"}
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _clean_link(m):
    text, url = m.group(1).strip(), m.group(2)
    if "{{" in url or url.startswith(("javascript:", "#", "?")) or url in ("/", ""):
        return text
    return m.group(0)


def _table_row_to_bullet(cells):
    """Turn a row of a header-less SharePoint list table into one readable bullet."""
    cells = [c.strip() for c in cells]
    if any("DispForm.aspx" in c for c in cells):          # photo-gallery rows
        return None
    title, url, rest = "", "", []
    for c in cells:
        if not c or c in DROP_CELLS:
            continue
        m = LINK.fullmatch(c)
        if m and not title:
            title, url = m.group(1).strip(), m.group(2)
            continue
        if re.fullmatch(r"https?://\S+", c):
            url = url or c
            continue
        # SharePoint taxonomy paths, bare file names and numeric ids add noise, not meaning
        if (c.startswith("עץ ") or re.fullmatch(r"\d{1,6}", c)
                or re.fullmatch(r"[\w.\- ]+\.(pdf|docx?|xlsx?|pptx?)", c, re.I)):
            continue
        c = re.sub(r"(\d{1,2}/\d{1,2}/\d{4}) 00:00:00", r"\1", c)
        c = re.sub(r"(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}):00", r"\1", c)
        if c == title or c in rest:
            continue
        if not title:
            title = c
            continue
        rest.append(c)
    if not title:
        return None
    head = f"[{title}]({url})" if url else title
    return "* " + " — ".join([head] + rest)


def _convert_tables(lines):
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s|:-]+\|\s*$", lines[i + 1]):
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(lines[j].strip().strip("|").split("|"))
                j += 1
            if any(header):                        # a real table: keep it as Markdown
                out += [ln, lines[i + 1]] + [lines[k] for k in range(i + 2, j)]
            else:                                  # SharePoint list dump: one bullet per row
                out += [b for b in (_table_row_to_bullet(r) for r in rows) if b]
            i = j
            continue
        out.append(ln)
        i += 1
    return out


def clean_page(text):
    text = text.replace("\\*", "*")
    # "* ### גיל\n\n  3"  ->  "* גיל: 3"   (SharePoint field label + value)
    text = re.sub(r"(?m)^(\s*)[*+]\s*(?:#{1,6}\s*)?([^\n:]{1,60}?):?\s*\n\s*\n\s{2,}(\S[^\n]*)$",
                  r"\1* \2: \3", text)
    text = LINK.sub(_clean_link, text)
    lines = _convert_tables(text.split("\n"))
    out, drop_level = [], 0
    for ln in lines:
        s = ln.strip()
        h = HEADING.match(s.lstrip("*+ ").strip()) if s.startswith(("#", "* #", "+ #")) else None
        if h and s.startswith("#"):
            level = len(h.group(1))
            if drop_level and level > drop_level:
                continue
            drop_level = level if DROP_SECTION.match(h.group(2)) else 0
            if drop_level or DROP_HEADING.match(h.group(2)):
                continue
        elif drop_level:
            continue
        if s in PLACEHOLDER or re.fullmatch(r"\[?>?Back to top\]?(\(#\))?|255|תוים\.|left", s):
            continue
        out.append(ln.rstrip())
    # drop headings with nothing under them (before the next heading of the same or higher level)
    keep = [True] * len(out)
    for i, ln in enumerate(out):
        m = HEADING.match(ln.strip())
        if not m:
            continue
        lvl = len(m.group(1))
        has_body = False
        for nxt in out[i + 1:]:
            if not nxt.strip():
                continue
            m2 = HEADING.match(nxt.strip())
            if m2 and len(m2.group(1)) <= lvl:
                break
            if not m2:
                has_body = True
                break
        keep[i] = has_body
    text = "\n".join(l for l, k in zip(out, keep) if k)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- document extraction

def _text_quality(t):
    """Share of letters among non-space chars; legacy-encoded Hebrew PDFs come out as symbols."""
    chars = [c for c in t if not c.isspace()]
    if not chars:
        return 0.0
    good = sum(1 for c in chars if ("֐" <= c <= "׿") or c.isascii() and c.isalnum()
               or ("Ѐ" <= c <= "ӿ"))
    return good / len(chars)


def _tidy(t):
    t = re.sub(r"[​‎‏﻿\xa0]", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    # PDF text breaks lines mid-sentence; rejoin, keep blank lines as paragraph breaks
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", t)]
    return "\n\n".join(p for p in paras if p)


# Word pairs that are common in municipal Hebrew; seeing them backwards means the PDF stores
# text in visual (left-to-right) order and every line's words come out reversed.
_PAIRS = [("עיריית", "נתניה"), ("על", "ידי"), ("ראש", "העיר"), ("בית", "ספר"), ("מועצת", "העיר"),
          ("חוק", "עזר"), ("לפי", "סעיף"), ("בהתאם", "לחוק"), ("סגן", "ראש"), ("חבר", "המועצה"),
          ("של", "העירייה"), ("את", "הדוח"), ("משרד", "הפנים"), ("מבקר", "העירייה"), ("כפי", "שנקבע")]
_LEAD_PUNCT = re.compile(r"^([.,:;!?]+)(?=[א-ת\"'])")
_MIRROR = str.maketrans("()[]", ")(][")


def _is_visual_order(text):
    words = re.findall(r"[א-ת]+", text[:200000])
    bigrams = collections.Counter(zip(words, words[1:]))
    fwd = sum(bigrams[p] for p in _PAIRS)
    rev = sum(bigrams[(b, a)] for a, b in _PAIRS)
    return rev >= 3 and rev > 2 * fwd


def fix_rtl(raw):
    """Repair Hebrew PDF text line by line (must run before lines are joined)."""
    visual = _is_visual_order(raw)
    out = []
    for line in raw.split("\n"):
        s = line.strip()
        if not re.search(r"[א-ת]", s):
            out.append(s)
            continue
        if visual:   # reverse word order, put punctuation back at the end of words, unmirror brackets
            toks = [t.translate(_MIRROR) for t in s.split()[::-1]]
            toks = [_LEAD_PUNCT.sub("", t) + (_LEAD_PUNCT.match(t).group(1) if _LEAD_PUNCT.match(t) else "")
                    for t in toks]
            s = " ".join(toks)
        else:        # logical order, but sentence-final punctuation landed at the start of the line
            m = _LEAD_PUNCT.match(s)
            if m:
                s = s[m.end():] + m.group(1)
        out.append(s)
    return "\n".join(out), visual


def extract_doc(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import pymupdf
        # No sort=True: sorting spans by position takes many minutes on map/blueprint PDFs
        # with thousands of labels. Stop after 300 pages or 60 s and keep what was read.
        t0 = time.time()
        with pymupdf.open(path) as d:
            parts = []
            for i, pg in enumerate(d):
                if i >= 300 or time.time() - t0 > 60:
                    break
                parts.append(pg.get_text("text"))
            text, _ = fix_rtl("\n\n".join(parts))
            return _tidy(text), d.page_count
    if ext == ".docx":
        import docx
        d = docx.Document(path)
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for tbl in d.tables:
            for row in tbl.rows:
                cells = []
                for c in row.cells:
                    if c.text.strip() and c.text.strip() not in cells:
                        cells.append(c.text.strip())
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts), None
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
            for n, row in enumerate(rows):
                if n > 5000:
                    break
                cells = ["" if c is None else str(c).strip() for c in row]
                if any(cells):
                    rows_out.append(" | ".join(cells).rstrip(" |"))
        return "\n".join(rows_out), None
    if ext == ".pptx":
        from pptx import Presentation
        parts = []
        for i, slide in enumerate(Presentation(path).slides, 1):
            parts.append(f"## Slide {i}")
            parts += [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame]
        return "\n".join(parts), None
    if ext == ".doc":     # legacy Word: pull UTF-16LE Hebrew/ASCII runs out of the binary
        data = open(path, "rb").read()
        runs = re.findall(rb"(?:[\x20-\x7e\r\n\t]\x00|[\xd0-\xea]\x05){4,}", data)
        return _tidy("\n".join(r.decode("utf-16-le", "ignore") for r in runs).replace("\r", "\n")), None
    return "", None


def extract_all_docs(data_dir, cache_path):
    docs = [json.loads(l) for l in open(os.path.join(data_dir, "documents.jsonl"), encoding="utf-8") if l.strip()]
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            r = json.loads(l)
            cache[r["file"]] = r
    todo = [d for d in docs if d["file"] not in cache]
    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as f:
        for i, d in enumerate(todo, 1):
            fp = os.path.join(data_dir, d["file"])
            rec = {"file": d["file"], "text": "", "pages": None, "status": "missing"}
            if os.path.exists(fp):
                try:
                    text, pages = extract_doc(fp)
                    text = text.strip()
                    q = _text_quality(text)
                    status = ("ok" if len(text) >= 80 and q >= 0.55 else
                              "no_text" if len(text) < 80 else "garbled")
                    rec = {"file": d["file"], "text": text if status == "ok" else "",
                           "pages": pages, "status": status, "quality": round(q, 2)}
                except Exception as e:
                    rec["status"] = "error"
                    rec["error"] = str(e)[:200]
            cache[d["file"]] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  extracted {i}/{len(todo)} documents ({time.time() - t0:.0f}s)", flush=True)
    return docs, cache


# ---------------------------------------------------------------- metadata

SECTION_NAMES = {
    "News": "חדשות", "City/Events": "אירועים", "City/Gardens": "גנים ופארקים",
    "City/SportsAndSociety": "ספורט וחברה", "Residents/Education": "חינוך",
    "Residents/Welfare": "רווחה", "Residents/Arnona": "ארנונה", "CityHall/Engineering": "הנדסה ותכנון",
    "City/SecurityAndEmergency": "ביטחון וחירום", "CityHall/transparency": "שקיפות ומכרזים",
    "CityHall/ServicesInnovation": "שירות ודיגיטל", "CityHall/Units": "יחידות העירייה",
    "Business/BLTS": "רישוי עסקים", "City/NB": "שכונות ורובעים",
    "Residents/VeterinaryService": "שירות וטרינרי", "City/ParkingAndTraffic": "חניה ותנועה",
    "City/RoadSafety": "בטיחות בדרכים", "City/Environment": "איכות הסביבה",
    "CityHall/MCN": "מורשת והנצחה", "City/CityAppearance": "שיפור פני העיר",
    "CityHall/QP": "איכות ותכנון אסטרטגי", "Residents/AliyahandIntegration": "עלייה וקליטה",
    "City/Beaches": "חופים", "CityHall/InspectionPolicing": "פיקוח ושיטור",
    "City/JewishCulture": "תרבות יהודית", "City/garbagedisposal": "פינוי אשפה",
    "City/PublicHealth": "בריאות הציבור", "City/Holdings": "החזקות עירוניות",
    "Business/BusinessDevelopment": "פיתוח עסקים", "sites/Business": "עסקים",
    "City/Infrastructure": "תשתיות", "en": "English", "ru": "Русский", "fr": "Français",
}


def url_section(url):
    parts = [p for p in unquote(urlparse(url).path).split("/") if p]
    if not parts:
        return "home"
    if parts[0].lower() in ("en", "ru", "fr"):
        return parts[0].lower()
    if parts[0].lower() == "news":
        return "News"
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def page_kind(url):
    u = unquote(url)
    for pat, kind in ((r"/News/Pages/(article|artice)", "news"), (r"/Events/Pages/event\d", "event"),
                      (r"/Gardens/", "garden"), (r"/facilities/", "facility"),
                      (r"/Education/.*/(Schools|Preschool|SEI|KG)", "school"), (r"/Tenders/", "tender"),
                      (r"/Units/", "unit")):
        if re.search(pat, u, re.I):
            return kind
    return "page"


# ---------------------------------------------------------------- labels (ported from muniflow_kb_v1)

# section_name -> one of 20 municipal domains
DOMAIN_MAP = {
    "ארנונה": "taxes_and_payments", "רישוי עסקים": "business_licensing", "פיתוח עסקים": "business_licensing",
    "עסקים": "business_licensing", "חינוך": "education", "גנים ופארקים": "parks_and_public_space",
    "הנדסה ותכנון": "planning_and_building", "תשתיות": "planning_and_building", "רווחה": "welfare",
    "איכות הסביבה": "environment_and_sanitation", "פינוי אשפה": "environment_and_sanitation",
    "חניה ותנועה": "transport_and_parking", "בטיחות בדרכים": "transport_and_parking",
    "ביטחון וחירום": "security_and_emergency", "שירות ודיגיטל": "resident_services",
    "יחידות העירייה": "municipal_administration", "איכות ותכנון אסטרטגי": "municipal_administration",
    "החזקות עירוניות": "municipal_administration", "שקיפות ומכרזים": "procurement_and_transparency",
    "פיקוח ושיטור": "enforcement", "שיפור פני העיר": "parks_and_public_space", "חופים": "parks_and_public_space",
    "שירות וטרינרי": "veterinary", "בריאות הציבור": "public_health", "עלייה וקליטה": "community_and_culture",
    "שכונות ורובעים": "community_and_culture", "ספורט וחברה": "community_and_culture",
    "מורשת והנצחה": "community_and_culture", "תרבות יהודית": "community_and_culture", "אירועים": "events",
    "חדשות": "news", "English": "multilingual", "Русский": "multilingual", "Français": "multilingual",
}
DOMAIN_HE = {
    "taxes_and_payments": "ארנונה ותשלומים", "business_licensing": "רישוי ועסקים", "education": "חינוך",
    "parks_and_public_space": "מרחב ציבורי, גנים וחופים", "planning_and_building": "הנדסה, תכנון ותשתיות",
    "welfare": "רווחה", "environment_and_sanitation": "סביבה ותברואה", "transport_and_parking": "חניה ותחבורה",
    "security_and_emergency": "ביטחון וחירום", "resident_services": "שירותים לתושב",
    "municipal_administration": "עירייה ומחלקות", "procurement_and_transparency": "מכרזים ושקיפות",
    "enforcement": "פיקוח ואכיפה", "veterinary": "שירות וטרינרי", "public_health": "בריאות הציבור",
    "community_and_culture": "קהילה, חברה ותרבות", "events": "אירועים", "news": "חדשות",
    "multilingual": "תוכן בשפות נוספות", "other": "אחר / דורש מיון",
}
TOPICS = {
    "property_tax_discount": ("הנחות בארנונה", r"הנח(?:ה|ות)\s+(?:ב|ל)?ארנונה|פטור\s+מארנונה"),
    "property_tax_payment": ("תשלום ארנונה", r"תשלום\s+ארנונה|תשלומי\s+ארנונה|חוב\s+ארנונה"),
    "property_holder_change": ("חילופי מחזיקים", r"חילופי\s+מחזיק|שינוי\s+מחזיק|החלפת\s+מחזיק"),
    "payment_arrangement": ("הסדר תשלומים", r"הסדר\s+תשלומים|פריסת\s+תשלומים"),
    "business_license": ("רישיון עסק", r"רישוי\s+עסקים|רישיון\s+עסק|רשיון\s+עסק"),
    "building_permit": ("היתר בנייה", r"היתר\s+בני[יה]|בקשה\s+להיתר"),
    "parking_permit": ("תו חניה", r"תו\s+חניה|תג\s+חניה|דוח\s+חניה"),
    "kindergarten_registration": ("רישום לגני ילדים", r"רישום\s+לג(?:ן|ני)\s+ילדים|הרשמה\s+לגנים"),
    "school_registration": ("רישום לבתי ספר", r"רישום\s+לבתי\s+ספר|רישום\s+לכיתה"),
    "social_services": ("שירותי רווחה", r"שירותי\s+רווחה|מחלקת\s+הרווחה|עובד\s+סוציאלי"),
    "resident_request_106": ("מוקד 106", r"מוקד\s*106|פניה\s+למוקד|פנייה\s+למוקד"),
    "waste_collection": ("פינוי פסולת", r"פינוי\s+(?:אשפה|פסולת|גזם)|איסוף\s+אשפה"),
    "public_shelter": ("מקלטים ציבוריים", r"מקלטים|מרחב\s+מוגן"),
}
TOPIC_RX = {k: re.compile(v[1]) for k, v in TOPICS.items()}
SERVICE_RE = re.compile(r"ארנונה|הנחה|תשלום|רישום|רישוי|בקשה|טופס|אישור|תעודה|דיווח|היתר|חניה|מחזיק|"
                        r"קבלת קהל|מוקד|הסדר|זכאות|פינוי|פנייה")
# Only these are hidden by search --no-archives. muniflow also hid "mixed_archive" (every page in the
# transparency section), which drops current pages such as the service charter and appeals committee.
TIME_SENSITIVE = ("news_time_sensitive", "events_time_sensitive", "procurement_time_sensitive")


def add_labels(s):
    """domain, category, review_status and topics. Labels are heuristics, not verified facts:
    `service_candidate_review` marks pages to check by hand, not confirmed service procedures."""
    k = s["kind"]
    s["domain"] = DOMAIN_MAP.get(s["section_name"], "other")
    s["domain_name"] = DOMAIN_HE[s["domain"]]
    s["category"] = ({"news": "news_time_sensitive", "event": "events_time_sensitive",
                      "tender": "procurement_time_sensitive"}.get(k)
                     or ("supporting_document" if k == "document" else
                         "directory_entry" if k in ("unit", "facility", "school", "garden") else
                         "mixed_archive" if s["section_name"] in ("חדשות", "אירועים", "שקיפות ומכרזים") else
                         "reference_page"))
    s["review_status"] = ("archive_refresh_before_use" if k in ("news", "event", "tender") else
                          "reference_check_document_date" if k == "document" else
                          "service_candidate_review" if k == "page" and SERVICE_RE.search(s["title"]) else
                          "general_reference")
    head = s["title"] + " " + s["text"][:1200]
    s["topics"] = [t for t, rx in TOPIC_RX.items() if rx.search(head)]
    return s


def norm_lang(l):
    l = (l or "").lower()
    return l[:2] if l[:2] in ("he", "en", "ru", "fr") else "he"


# ---------------------------------------------------------------- chunking

def split_sections(text):
    """Yield (heading_path, body) for each heading-delimited section."""
    path, body = [], []
    for ln in text.split("\n"):
        m = HEADING.match(ln.strip())
        if m:
            if "\n".join(body).strip():
                yield list(path), "\n".join(body).strip()
            body = []
            lvl = len(m.group(1))
            path = [p for p in path if p[0] < lvl] + [(lvl, m.group(2))]
        else:
            body.append(ln)
    if "\n".join(body).strip():
        yield list(path), "\n".join(body).strip()


def _split_long(s, target, hard):
    """Split one over-long block at paragraph, line, then sentence boundaries."""
    if len(s) <= hard:
        return [s]
    for sep in ("\n\n", "\n", ". ", " "):
        pieces = s.split(sep)
        if len(pieces) > 1:
            break
    out, cur = [], ""
    for p in pieces:
        cand = cur + sep + p if cur else p
        if len(cand) > target and cur:
            out.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        out.append(cur)
    return [x for o in out for x in (_split_long(o, target, hard) if len(o) > hard else [o])]


def chunk_text(text, target=1200, overlap=150):
    hard = int(target * 1.6)
    units = []                                   # (heading string, block)
    for path, body in split_sections(text):
        head = " > ".join(h for _, h in path)
        for block in _split_long(body, target, hard):
            units.append((head, block))
    chunks, cur_head, cur = [], None, ""
    for head, block in units:
        if cur and (len(cur) + len(block) > target or (head != cur_head and len(cur) > target * 0.5)):
            chunks.append((cur_head, cur))
            tail = cur[-overlap:] if head == cur_head and overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = (tail + "\n" + block) if tail else block
            cur_head = head
        else:
            if cur and head != cur_head and head:
                block = f"### {head.split(' > ')[-1]}\n{block}"
            cur = (cur + "\n\n" + block) if cur else block
            cur_head = cur_head if cur_head is not None else head
    if cur.strip():
        chunks.append((cur_head, cur))
    return [(h or "", c.strip()) for h, c in chunks if len(c.strip()) >= 40]


# ---------------------------------------------------------------- full-text search helpers

HEB_PREFIX = re.compile(r"^(?:ו?(?:ש|כש|מ|ה|ב|ל|כ|לכש|מה|שה|שב|של|וה|וב|ול|ומ|וכ))")
NIQQUD = re.compile(r"[֑-ׇ]")


def fold_terms(text):
    """Extra index terms: Hebrew words with their one-letter prefixes (ו ה ב ל מ ש כ) removed,
    so 'בארנונה' / 'והארנונה' also match 'ארנונה'."""
    text = NIQQUD.sub("", text)
    extra = set()
    for w in re.findall(r"[א-ת]{4,}", text):
        m = HEB_PREFIX.match(w)
        if m and len(w) - m.end() >= 3:
            extra.add(w[m.end():])
    return " ".join(sorted(extra))


def build_sqlite(path, sources, chunks):
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE sources(id TEXT PRIMARY KEY, kind TEXT, url TEXT, title TEXT, lang TEXT,
                             section TEXT, section_name TEXT, breadcrumb TEXT, source_page TEXT,
                             file TEXT, domain TEXT, category TEXT, review_status TEXT, topics TEXT,
                             text TEXT);
        CREATE TABLE chunks(id TEXT PRIMARY KEY, source_id TEXT, ord INTEGER, url TEXT, title TEXT,
                            heading TEXT, lang TEXT, section TEXT, kind TEXT, domain TEXT,
                            category TEXT, review_status TEXT, topics TEXT, content TEXT, text TEXT);
        CREATE TABLE domains(id TEXT PRIMARY KEY, name_he TEXT);
        CREATE TABLE topics(id TEXT PRIMARY KEY, name_he TEXT, pattern TEXT);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(title, heading, content, folded,
                            tokenize = "unicode61 remove_diacritics 2");
    """)
    db.executemany("INSERT INTO domains VALUES (?,?)", DOMAIN_HE.items())
    db.executemany("INSERT INTO topics VALUES (?,?,?)", [(k, v[0], v[1]) for k, v in TOPICS.items()])
    db.executemany("INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (s["id"], s["kind"], s["url"], s["title"], s["lang"], s["section"], s["section_name"],
         " > ".join(s.get("breadcrumb") or []), s.get("source_page"), s.get("file"), s["domain"],
         s["category"], s["review_status"], ",".join(s["topics"]), s["text"]) for s in sources])
    db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (c["id"], c["source_id"], c["ord"], c["url"], c["title"], c["heading"], c["lang"],
         c["section"], c["kind"], c["domain"], c["category"], c["review_status"],
         ",".join(c["topics"]), c["content"], c["text"]) for c in chunks])
    db.execute("CREATE INDEX chunks_domain ON chunks(domain)")
    db.executemany("INSERT INTO chunks_fts(rowid, title, heading, content, folded) VALUES (?,?,?,?,?)", [
        (i, c["title"], c["heading"], c["content"],
         fold_terms(c["title"] + " " + c["heading"] + " " + c["content"]))
        for i, c in enumerate(chunks, 1)])
    db.execute("CREATE TABLE chunk_rowid(rowid INTEGER PRIMARY KEY, chunk_id TEXT)")
    db.executemany("INSERT INTO chunk_rowid VALUES (?,?)", [(i, c["id"]) for i, c in enumerate(chunks, 1)])
    db.commit()
    db.execute("VACUUM")
    db.close()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("rag-data", "netanya"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=int, default=1200, help="target chunk size in characters")
    ap.add_argument("--export", default=None,
                    help="single-file JSONL of all chunks (default: <data>/../netanya_knowledge_base.jsonl)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = a.out or os.path.join(a.data, "kb")
    os.makedirs(out, exist_ok=True)

    print("extracting documents ...", flush=True)
    docs, doc_text = extract_all_docs(a.data, os.path.join(out, "doc_text.jsonl"))

    pages = [json.loads(l) for l in open(os.path.join(a.data, "pages.jsonl"), encoding="utf-8") if l.strip()]
    title_by_url = {p["url"]: p.get("title", "") for p in pages}
    sources, raw_chars, clean_chars = [], 0, 0
    seen_text = set()
    stats = collections.Counter()
    for p in pages:
        raw_chars += len(p["text"])
        text = redact(clean_page(p["text"]))
        desc = (p.get("description") or "").strip()
        if desc and desc[:60] not in text and not desc.startswith("באתר עיריית נתניה תוכלו"):
            text = desc + "\n\n" + text
        h = hashlib.sha1(text.encode()).hexdigest()
        if len(text) < 60 or h in seen_text:
            stats["pages_dropped_empty_or_dup"] += 1
            continue
        seen_text.add(h)
        clean_chars += len(text)
        sec = url_section(p["url"])
        sources.append({
            "id": "p_" + hashlib.sha1(p["url"].encode()).hexdigest()[:12], "kind": page_kind(p["url"]),
            "url": p["url"], "title": p.get("title") or "", "lang": norm_lang(p.get("lang")),
            "section": sec, "section_name": SECTION_NAMES.get(sec, sec),
            "breadcrumb": [b for b in (p.get("breadcrumb") or [])
                           if b not in ("האתר הרשמי של עיריית נתניה", "Netanya Municipality")],
            "tags": p.get("tags") or [], "fetched_at": p.get("fetched_at"), "text": text})
    doc_status = collections.Counter()
    for d in docs:
        rec = doc_text.get(d["file"], {})
        doc_status[rec.get("status", "missing")] += 1
        text = redact(rec.get("text") or "")
        if rec.get("status") != "ok":
            continue
        h = hashlib.sha1(text.encode()).hexdigest()
        if h in seen_text:
            doc_status["duplicate"] += 1
            continue
        seen_text.add(h)
        name = os.path.basename(unquote(urlparse(d["url"]).path))
        parent = title_by_url.get(d.get("source_page"), "")
        sec = url_section(d.get("source_page") or "")
        sources.append({
            "id": "d_" + hashlib.sha1(d["url"].encode()).hexdigest()[:12], "kind": "document",
            "url": d["url"], "title": f"{parent} — {name}" if parent else name,
            "lang": "he", "section": sec, "section_name": SECTION_NAMES.get(sec, sec),
            "breadcrumb": [parent] if parent else [], "tags": [], "source_page": d.get("source_page"),
            "file": d["file"], "pages": rec.get("pages"), "fetched_at": d.get("fetched_at"), "text": text})

    for s in sources:
        add_labels(s)

    chunks = []
    for s in sources:
        crumbs = " > ".join(s["breadcrumb"])
        for i, (head, content) in enumerate(chunk_text(s["text"], a.target)):
            ctx = [s["title"]] + ([crumbs] if crumbs and crumbs != s["title"] else []) + ([head] if head else [])
            chunks.append({
                "id": f"{s['id']}#{i}", "source_id": s["id"], "ord": i, "url": s["url"],
                "title": s["title"], "heading": head, "lang": s["lang"], "section": s["section"],
                "section_name": s["section_name"], "kind": s["kind"], "domain": s["domain"],
                "category": s["category"], "review_status": s["review_status"], "topics": s["topics"],
                "content": content,
                "text": "\n".join(ctx) + "\n\n" + content})

    for name, recs in (("sources.jsonl", sources), ("chunks.jsonl", chunks)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # the single-file deliverable: one chunk per line, embedding-ready text plus citation metadata
    export = a.export or os.path.join(os.path.dirname(os.path.normpath(a.data)) or ".",
                                      "netanya_knowledge_base.jsonl")
    with open(export, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({k: c[k] for k in ("id", "url", "title", "heading", "section_name",
                                                   "domain", "kind", "category", "review_status",
                                                   "topics", "lang", "text")}, ensure_ascii=False) + "\n")
    build_sqlite(os.path.join(out, "kb.sqlite"), sources, chunks)
    write_report(os.path.join(out, "REPORT.md"), pages, docs, doc_text, sources, chunks,
                 raw_chars, clean_chars, stats, doc_status)
    print(f"done: {len(sources)} sources, {len(chunks)} chunks -> {out}")


def write_report(path, pages, docs, doc_text, sources, chunks, raw_chars, clean_chars, stats, doc_status):
    by_sec = collections.Counter((s["section_name"], s["kind"] == "document") for s in sources)
    ch_sec = collections.Counter(c["section_name"] for c in chunks)
    secs = sorted({k for k, _ in by_sec}, key=lambda k: -ch_sec[k])
    kinds = collections.Counter(s["kind"] for s in sources)
    langs = collections.Counter(c["lang"] for c in chunks)
    sizes = sorted(len(c["content"]) for c in chunks)
    ext_status = collections.Counter((os.path.splitext(d["file"])[1].lower(), doc_text.get(d["file"], {}).get("status"))
                                     for d in docs)
    L = ["# Netanya municipality knowledge base", "",
         "Built by `scripts/build_netanya_kb.py` from the www.netanya.muni.il crawl.", "",
         "## Totals", "",
         f"- Sources: **{len(sources):,}** ({sum(1 for s in sources if s['kind'] != 'document'):,} web pages, "
         f"{kinds['document']:,} documents)",
         f"- Chunks: **{len(chunks):,}** — median {sizes[len(sizes) // 2]:,} chars, p95 {sizes[int(len(sizes) * .95)]:,}",
         f"- Page text: {raw_chars:,} raw chars → {clean_chars:,} after removing site boilerplate "
         f"({100 - clean_chars * 100 // max(raw_chars, 1)}% removed); "
         f"{stats['pages_dropped_empty_or_dup']} pages dropped as empty or duplicate",
         f"- Chunk languages: " + ", ".join(f"{k} {v:,}" for k, v in langs.most_common()),
         f"- Source kinds: " + ", ".join(f"{k} {v:,}" for k, v in kinds.most_common()), "",
         "## Coverage by section", "", "| Section | Pages | Documents | Chunks |", "| --- | ---: | ---: | ---: |"]
    for k in secs:
        L.append(f"| {k} | {by_sec[(k, False)]} | {by_sec[(k, True)]} | {ch_sec[k]} |")
    dom = collections.Counter(s["domain"] for s in sources)
    dom_ch = collections.Counter(c["domain"] for c in chunks)
    L += ["", "## Domains", "", "| Domain | Name | Sources | Chunks |", "| --- | --- | ---: | ---: |"]
    L += [f"| `{d}` | {DOMAIN_HE[d]} | {n} | {dom_ch[d]} |" for d, n in dom.most_common()]
    rev = collections.Counter(s["review_status"] for s in sources)
    cat = collections.Counter(s["category"] for s in sources)
    top = collections.Counter(t for s in sources for t in s["topics"])
    L += ["", "## Review flags and categories (sources)", "",
          "Heuristic labels to guide review, not verified facts. `service_candidate_review` pages are "
          "candidates for a hand-checked service list.", ""]
    L += [f"- `{k}`: {v:,}" for k, v in rev.most_common()]
    L += [""] + [f"- category `{k}`: {v:,}" for k, v in cat.most_common()]
    L += ["", "## Topics (sources)", ""] + [f"- {TOPICS[k][0]} (`{k}`): {v}" for k, v in top.most_common()]
    L += ["", "## Documents (PDF / Office attachments)", "",
          "| Type | Usable text | No text (scanned/image) | Garbled (legacy font encoding) | Error |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for ext in sorted({e for e, _ in ext_status}):
        L.append(f"| {ext} | {ext_status[(ext, 'ok')]} | {ext_status[(ext, 'no_text')]} | "
                 f"{ext_status[(ext, 'garbled')]} | {ext_status[(ext, 'error')]} |")
    L += ["", f"Duplicate document texts skipped: {doc_status['duplicate']}.", "",
          "Documents with no or garbled text need OCR (e.g. Tesseract with the `heb` model) to be included.",
          "", "## Known limitations", "",
          "- Hebrew PDFs are extracted in logical order, but some brackets/punctuation come out mirrored "
          "and table layouts are flattened.",
          "- News pages carry no publish date in the crawled text; `fetched_at` is the crawl time.",
          "- Event, tender and news content is time-sensitive and goes stale; re-crawl and rebuild regularly.",
          "- The index is lexical (SQLite FTS5/BM25 with Hebrew prefix folding). `chunks.jsonl` `text` is "
          "ready for embedding if semantic search is added.", ""]
    open(path, "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
