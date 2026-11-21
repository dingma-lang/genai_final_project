"""Mask Guardian match previews at five cumulative levels.

    L1  original text (whitespace normalized only)
    L2  + home and away team names and nicknames -> [HOME] / [AWAY]
    L3  + journalist byline removed, years -> [YEAR], calendar dates -> [DATE]
    L4  + other clubs -> [CLUB_n], people -> [PERSON_n], stadiums -> [VENUE],
          places -> [PLACE], nationalities -> [NATIONALITY], other proper names -> [NAME_n]
    L5  + time and season cues: weekdays, months, holidays, "opening day", "final day",
          summer/winter, dated tournaments -> [TIME] / [EVENT]; sponsored cup names -> "League Cup"

Every candidate name is found once in the original text, overlaps are resolved once
(longest match wins), and each level applies the categories switched on at that level.
So each level masks everything the level below it masks. Numbering ([PERSON_1] ...) is
consistent within a preview and across levels: the same surname always gets the same number.

Usage:
    python mask_previews.py <previews_with_results.csv> <output_dir>

Writes to <output_dir>:
    llm_inputs_L1.jsonl ... llm_inputs_L5.jsonl   {"MatchID", "text", "input_sha256"} per line,
                                                  the format the extraction scripts read
    masked_previews.csv     one row per preview with all five levels side by side
    masking_log.csv         every replacement: MatchID, level, category, original, replacement
    person_lexicon.csv      names found as people across the corpus and reused to catch misses
Requires: pandas, spacy, and the spaCy model en_core_web_lg.
"""
import collections
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from masking_lists import (CLUB_ALIASES, DATED_EVENTS, HOLIDAYS, KEEP, MANUAL_PERSONS, MONTHS,  # noqa: E402
                           NATIONALITY_NOUNS,
                           OTHER_CLUBS, PLACES, SEASON_PHRASES, SHARED_NICKNAMES, SPONSORED_CUPS, VENUES,
                           WEEKDAYS)

LEVEL = {"team": 2, "byline": 3, "year": 3, "date": 3, "club": 4, "venue": 4, "place": 4, "person": 4,
         "nationality": 4, "name": 4, "time": 5, "event": 5, "cup": 5}
# Tie-break when two candidate spans have the same length (lower = preferred).
PRIORITY = {"team": 0, "club": 1, "venue": 2, "place": 3, "person": 4, "byline": 5, "date": 6, "year": 7,
            "cup": 8, "event": 9, "time": 10, "nationality": 11, "name": 12}
APOS = "’'"
POSSESSIVE_INSIDE = re.compile(r"[’']s\b")
SENTENCE_START = re.compile(r"(?:^|[.!?:;]\s+|[\"“]\s*)$")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_pattern(terms, flags=0):
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    return re.compile(r"(?<![\w&-])(?:" + "|".join(re.escape(t) for t in terms) + r")(?![\w&])", flags)


def strip_possessive(start, end, text):
    if end - start > 2 and text[end - 2] in APOS and text[end - 1] == "s":
        end -= 2
    elif end - start > 1 and text[end - 1] in APOS:
        end -= 1
    return start, end


def at_sentence_start(text, start):
    return bool(SENTENCE_START.search(text[:start]))


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
ALL_SAMPLE_ALIASES = {a: club for club, aliases in CLUB_ALIASES.items() for a in aliases}
NICK_OWNERS = collections.defaultdict(set)
for club, aliases in CLUB_ALIASES.items():
    for a in aliases:
        NICK_OWNERS[a].add(club)
NON_PERSON_TERMS = (set(ALL_SAMPLE_ALIASES) | set(OTHER_CLUBS) | set(VENUES) | set(PLACES) | KEEP
                    | set(NATIONALITY_NOUNS)
                    | set(WEEKDAYS) | set(MONTHS) | {"May"} | set(HOLIDAYS) | set(SPONSORED_CUPS) | set(DATED_EVENTS))

BYLINE = re.compile(r"(?<=[.!?”\"])\s+([A-ZÀ-Ý][\wÀ-ÿ’'-]+(?:\s+[A-ZÀ-Ý][\wÀ-ÿ’'-]+){1,3})\s*$")
YEAR = re.compile(r"(?<![\w£$€])(?:18|19|20)\d{2}(?:-\d{2})?(?![\w%])")
# Numeric dates use slashes or dots only, so formations ("4-2-3-1") and scores ("2-1") are untouched.
NUM_DATE = re.compile(r"(?<![\d./-])\d{1,2}[/.]\d{1,2}(?:[/.]\d{2,4})?(?![\d/-])")
ALL_MONTHS = MONTHS + ["May"]
DAY_MONTH = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?(?:\s+of)?\s+(?:" + "|".join(ALL_MONTHS) + r")\b"
                       r"|\b(?:" + "|".join(ALL_MONTHS) + r")\s+\d{1,2}(?:st|nd|rd|th)?\b")
# Time words may follow a hyphen ("mid-March", "pre-Christmas").
TIME_WORDS = re.compile(r"(?<![\w&])(?:" + "|".join(re.escape(t) for t in sorted(WEEKDAYS + MONTHS + HOLIDAYS, key=len, reverse=True)) + r")(?![\w&])")
YEAR_SEASONS = re.compile(r"(?<![\w-])(?:summer|winter|spring|autumn)(?![\w-])", re.IGNORECASE)
MAY = re.compile(r"(?<!\w)May(?!\w)")
SEASON = word_pattern(SEASON_PHRASES, re.IGNORECASE)
CUPS = word_pattern(SPONSORED_CUPS)
EVENTS = word_pattern(DATED_EVENTS)
VENUE_RE = word_pattern(VENUES)
PLACE_RE = word_pattern(PLACES)
NATION_RE = word_pattern(NATIONALITY_NOUNS)


def build_person_lexicon(docs):
    """Names spaCy tags as people anywhere in the corpus, reused to catch mentions it misses."""
    full, single = collections.Counter(), collections.Counter()
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            s, e = strip_possessive(ent.start_char, ent.end_char, doc.text)
            name = doc.text[s:e].strip()
            if not name or name in NON_PERSON_TERMS or not name[0].isupper() or "[" in name:
                continue
            words = name.split()
            # Reject spaCy errors such as "Will Chelsea" or "Steve Bruce’s Hull". Surnames that are
            # also foreign club names ("Enner Valencia") are allowed.
            if any(w in NON_PERSON_TERMS - set(OTHER_CLUBS) for w in words) or POSSESSIVE_INSIDE.search(name):
                continue
            if len(words) > 1:
                full[name] += 1
                last = words[-1]
                if last not in NON_PERSON_TERMS and last[0].isupper() and len(last) > 2:
                    single[last] += 1
            else:
                single[name] += 1
    # A single word counts as a person's name if spaCy saw it as part of a name at least twice.
    singles = {w for w, n in single.items() if n >= 2 and len(w) > 2}
    manual = set(MANUAL_PERSONS)
    return set(full) | {m for m in manual if " " in m}, singles | {m for m in manual if " " not in m}


def find_spans(text, home, away, doc, full_names, surnames, lowercase_vocab):
    """Return non-overlapping (start, end, category, key) spans for one preview."""
    cands = []

    def add(s, e, cat, key=None):
        if e > s:
            cands.append((s, e, cat, key if key is not None else text[s:e]))

    # Home/away names, other sample clubs, and shared nicknames.
    home_aliases, away_aliases = set(CLUB_ALIASES[home]), set(CLUB_ALIASES[away])
    involved = {home, away}
    for m in word_pattern(ALL_SAMPLE_ALIASES.keys() | set(OTHER_CLUBS)).finditer(text):
        t = m.group()
        owners = NICK_OWNERS.get(t, set())
        in_home, in_away = t in home_aliases, t in away_aliases
        if in_home and not in_away and (t not in SHARED_NICKNAMES or len(owners & involved) == 1):
            add(m.start(), m.end(), "team", "[HOME]")
        elif in_away and not in_home and (t not in SHARED_NICKNAMES or len(owners & involved) == 1):
            add(m.start(), m.end(), "team", "[AWAY]")
        elif in_home and in_away:
            add(m.start(), m.end(), "club", "TEAM")          # ambiguous between the two teams
        else:
            canon = ALL_SAMPLE_ALIASES.get(t, t) if t not in SHARED_NICKNAMES else t
            add(m.start(), m.end(), "club", "CLUB:" + canon)

    for m in VENUE_RE.finditer(text):
        add(m.start(), m.end(), "venue", "[VENUE]")
    for m in PLACE_RE.finditer(text):
        add(m.start(), m.end(), "place", "[PLACE]")
    for m in NATION_RE.finditer(text):
        add(m.start(), m.end(), "nationality", "[NATIONALITY]")
    for m in word_pattern(full_names).finditer(text) if full_names else []:
        add(m.start(), m.end(), "person", "PERSON:" + m.group().split()[-1].lower())
    # Surnames of people named in this preview, so a later lone "Wenger" matches "Arsène Wenger".
    local = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            s0, e0 = strip_possessive(ent.start_char, ent.end_char, text)
            words = text[s0:e0].split()
            if len(words) > 1 and words[-1] not in NON_PERSON_TERMS and words[-1][:1].isupper() and len(words[-1]) > 2:
                local.add(words[-1])
    for m in word_pattern(local).finditer(text) if local else []:
        add(m.start(), m.end(), "person", "PERSON:" + m.group().lower())
    for m in word_pattern(surnames).finditer(text) if surnames else []:
        w = m.group()
        if w.lower() in lowercase_vocab and at_sentence_start(text, m.start()):
            continue  # e.g. "Young players ..." at the start of a sentence
        add(m.start(), m.end(), "person", "PERSON:" + w.lower())

    # Dates and times.
    b = BYLINE.search(text)
    if b:
        add(b.start(), len(text), "byline", "")
    for m in YEAR.finditer(text):
        add(m.start(), m.end(), "year", "[YEAR]")
    for rx in (NUM_DATE, DAY_MONTH):
        for m in rx.finditer(text):
            add(m.start(), m.end(), "date", "[DATE]")
    for m in TIME_WORDS.finditer(text):
        add(m.start(), m.end(), "time", "[TIME]")
    for m in YEAR_SEASONS.finditer(text):
        add(m.start(), m.end(), "time", "[TIME]")
    for m in MAY.finditer(text):
        if not at_sentence_start(text, m.start()):
            add(m.start(), m.end(), "time", "[TIME]")
    for m in SEASON.finditer(text):
        add(m.start(), m.end(), "time", "[TIME]")
    for m in CUPS.finditer(text):
        add(m.start(), m.end(), "cup", "League Cup")
    for m in EVENTS.finditer(text):
        add(m.start(), m.end(), "event", "[EVENT]")

    # spaCy entities are added after the list matches.
    ner = []
    for ent in doc.ents:
        s, e = ent.start_char, ent.end_char
        lead = re.match(r"(?:the|The)\s+", text[s:e])
        if lead:
            s += lead.end()
        s, e = strip_possessive(s, e, text)
        while e > s and text[e - 1] in ".,;:!?)”\"":
            e -= 1
        name = text[s:e]
        if (e <= s or name in NON_PERSON_TERMS or name in KEEP or not name[:1].isupper()
                or any(k.lower() in name.lower() for k in KEEP if len(k) > 3)):   # e.g. "Premier League era"
            continue
        last = name.split()[-1]
        if ent.label_ == "PERSON" or name in full_names or last in surnames:
            # spaCy sometimes labels players as places or organisations ("Rio Ferdinand", "Diego Costa").
            ner.append((s, e, "person", "PERSON:" + name.split()[-1].lower()))
        elif ent.label_ in ("GPE", "LOC"):
            ner.append((s, e, "place", "[PLACE]"))
        elif ent.label_ == "FAC":
            ner.append((s, e, "venue", "[VENUE]"))
        elif ent.label_ == "NORP" and not name.startswith("Europe"):
            ner.append((s, e, "nationality", "[NATIONALITY]"))
        elif ent.label_ == "ORG":
            ner.append((s, e, "name", "NAME:" + name))
    spans = resolve([c + (0,) for c in cands] + [c + (1,) for c in ner])
    return extend_first_names(spans, text, lowercase_vocab)


def resolve(cands):
    """Choose non-overlapping spans. Lower masking levels win first, so every level masks
    everything the level below it masks; then list matches beat spaCy; then longer spans
    beat shorter ones; then category priority breaks ties."""
    out = []
    order = sorted(cands, key=lambda c: (LEVEL[c[2]], c[4], -(c[1] - c[0]), PRIORITY[c[2]], c[0]))
    for s, e, cat, key, _ in order:
        if all(e <= os or s >= oe for os, oe, _, _ in out):
            out.append((s, e, cat, key))
    return sorted(out)


def extend_first_names(spans, text, lowercase_vocab):
    """Absorb a capitalised first name left in front of a masked surname ("Salomon Rondón")."""
    out = []
    starts = {s for s, _, _, _ in spans}
    ends = {e for _, e, _, _ in spans}
    for s, e, cat, key in spans:
        if cat == "person":
            m = re.search(r"([A-ZÀ-Ý][\wÀ-ÿ’'-]*)\s+$", text[:s])
            if m and m.start(1) not in starts and m.end(1) not in ends:
                word = m.group(1)
                if (word not in NON_PERSON_TERMS and word.lower() not in lowercase_vocab
                        and not POSSESSIVE_INSIDE.search(word)):
                    s = m.start(1)
        out.append((s, e, cat, key))
    return out


def render(text, spans, level, numbering):
    pieces, log, pos = [], [], 0
    for s, e, cat, key in spans:
        if LEVEL[cat] > level:
            continue
        if key.startswith(("PERSON:", "CLUB:", "NAME:")):
            kind = key.split(":")[0]
            rep = f"[{kind}_{numbering[key]}]"
        elif key == "TEAM":
            rep = "[TEAM]"
        else:
            rep = key
        pieces.append(text[pos:s])
        pieces.append(rep)
        log.append((LEVEL[cat], cat, text[s:e], rep))
        pos = e
    pieces.append(text[pos:])
    out = "".join(pieces)
    out = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(out.split())).strip()
    return out, log


def main():
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values(["MatchDate", "MatchID"]).reset_index(drop=True)  # same order as the extraction pipeline
    unknown = (set(df.HomeStd) | set(df.AwayStd)) - set(CLUB_ALIASES)
    if unknown:
        sys.exit(f"Clubs missing from CLUB_ALIASES: {sorted(unknown)}")

    texts = [normalized(t) for t in df.Text]
    nlp = spacy.load("en_core_web_lg")
    docs = list(nlp.pipe(texts, batch_size=64))
    full_names, surnames = build_person_lexicon(docs)
    lowercase_vocab = {w for t in texts for w in re.findall(r"\b[a-z][a-z’'-]+\b", t)}
    pd.DataFrame({"name": sorted(full_names | surnames),
                  "kind": ["full name" if n in full_names else "single word" for n in sorted(full_names | surnames)]}
                 ).to_csv(out_dir / "person_lexicon.csv", index=False)

    levels = {k: [] for k in range(1, 6)}
    wide, log_rows = [], []
    for row, text, doc in zip(df.itertuples(), texts, docs):
        spans = find_spans(text, row.HomeStd, row.AwayStd, doc, full_names, surnames, lowercase_vocab)
        numbering, counters = {}, collections.Counter()
        for _, _, _, key in spans:
            if key.startswith(("PERSON:", "CLUB:", "NAME:")) and key not in numbering:
                kind = key.split(":")[0]
                counters[kind] += 1
                numbering[key] = counters[kind]
        rec = {"MatchID": int(row.MatchID), "Season": row.Season, "Home": row.HomeStd, "Away": row.AwayStd}
        for level in range(1, 6):
            masked, log = render(text, spans, level, numbering)
            rec[f"L{level}"] = masked
            levels[level].append({"MatchID": int(row.MatchID), "text": masked, "input_sha256": sha256(masked)})
            if level == 5:
                for lvl, cat, orig, rep in log:
                    log_rows.append({"MatchID": int(row.MatchID), "level": lvl, "category": cat,
                                     "original": orig, "replacement": rep})
        wide.append(rec)

    for level, records in levels.items():
        (out_dir / f"llm_inputs_L{level}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    pd.DataFrame(wide).to_csv(out_dir / "masked_previews.csv", index=False)
    log = pd.DataFrame(log_rows)
    log.to_csv(out_dir / "masking_log.csv", index=False)
    print(f"Masked {len(df)} previews at 5 levels -> {out_dir}")
    print("Replacements by level and category:")
    print(log.groupby(["level", "category"]).size().to_string())


if __name__ == "__main__":
    main()
