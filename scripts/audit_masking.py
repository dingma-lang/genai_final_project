"""Check what identifying information survives at each masking level.

Usage:
    python audit_masking.py <masked_previews.csv> <report_dir>

Two kinds of checks, both run on the masked text only:
  1. Known lists: does any team name, other club, stadium, place, known person, year,
     date or time word from the reference lists still appear?
  2. Independent check: spaCy is run again on the Level 4 and Level 5 text, and every
     remaining person, organisation, place or venue it finds is counted. This catches
     names that are in none of the lists.
Writes residual_counts.csv (share of previews with each kind of leftover, by level)
and leftover_names_L4.csv / leftover_names_L5.csv (what spaCy still finds, most common first).
"""
import collections
import re
import sys
from pathlib import Path

import pandas as pd
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from masking_lists import CLUB_ALIASES, KEEP, MONTHS, OTHER_CLUBS, PLACES, VENUES, WEEKDAYS  # noqa: E402
from mask_previews import (YEAR, NUM_DATE, DAY_MONTH, word_pattern, SHARED_NICKNAMES,  # noqa: E402
                           at_sentence_start, normalized)

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(src)
lexicon = pd.read_csv(src.parent / "person_lexicon.csv")
persons = word_pattern(set(lexicon.name))
vocab = {w for t in df.L1 for w in re.findall(r"\b[a-z][a-z’'-]+\b", normalized(t))}
PLACEHOLDERS = {"HOME", "AWAY", "TEAM", "VENUE", "PLACE", "NATIONALITY", "YEAR", "DATE", "TIME", "EVENT"}


def person_left(t):
    # A capitalised common word at the start of a sentence ("Given that ...") is not a name.
    return any(not (m.group().lower() in vocab and at_sentence_start(t, m.start())) for m in persons.finditer(t))

other_rx = word_pattern(OTHER_CLUBS)
venue_rx, place_rx = word_pattern(VENUES), word_pattern(PLACES)
time_rx = re.compile(r"(?<![\w&])(?:" + "|".join(WEEKDAYS + MONTHS) + r")(?![\w&])")


def own_team_names(row):
    names = set(CLUB_ALIASES[row.Home]) | set(CLUB_ALIASES[row.Away])
    return word_pattern(names - SHARED_NICKNAMES)


rows = []
for level in range(1, 6):
    col = f"L{level}"
    checks = collections.Counter()
    for r in df.itertuples():
        t = getattr(r, col)
        checks["home/away team name"] += bool(own_team_names(r).search(t))
        others = set().union(*[set(v) for k, v in CLUB_ALIASES.items() if k not in (r.Home, r.Away)])
        checks["other club"] += bool(word_pattern(others - SHARED_NICKNAMES).search(t) or other_rx.search(t))
        checks["stadium"] += bool(venue_rx.search(t))
        checks["place"] += bool(place_rx.search(t))
        checks["known person"] += person_left(t)
        checks["year"] += bool(YEAR.search(t))
        checks["calendar date"] += bool(NUM_DATE.search(t) or DAY_MONTH.search(t))
        checks["weekday or month"] += bool(time_rx.search(t))
    for k, v in checks.items():
        rows.append({"level": level, "check": k, "previews_with_leftover": v, "share": round(v / len(df), 3)})
counts = pd.DataFrame(rows).pivot(index="check", columns="level", values="share")
counts.to_csv(out / "residual_counts.csv")
print("Share of previews where each kind of information is still present, by level:")
print(counts.to_string())

nlp = spacy.load("en_core_web_lg")
for level in (4, 5):
    found = collections.Counter()
    examples = {}
    texts = df[f"L{level}"].tolist()
    for mid, doc in zip(df.MatchID, nlp.pipe(texts, batch_size=64)):
        for e in doc.ents:
            if e.label_ not in ("PERSON", "ORG", "GPE", "FAC", "LOC", "NORP", "EVENT"):
                continue
            name = re.sub(r"[’']s$", "", e.text.strip())
            name = re.sub(r"^(?:the|The)\s+", "", name)
            if ("[" in name or name in KEEP or not name[:1].isupper()
                    or re.sub(r"_\d+$", "", name) in PLACEHOLDERS):
                continue
            found[(name, e.label_)] += 1
            examples.setdefault((name, e.label_), mid)
    leftovers = pd.DataFrame([{"name": n, "spacy_label": lab, "count": c, "example_MatchID": examples[(n, lab)]}
                              for (n, lab), c in found.most_common()])
    leftovers.to_csv(out / f"leftover_names_L{level}.csv", index=False)
    print(f"\nLevel {level}: spaCy still finds {len(leftovers)} distinct names ({leftovers['count'].sum() if len(leftovers) else 0} mentions). Most common:")
    print(", ".join(f"{r.name} [{r.spacy_label}] {r.count}" for r in leftovers.head(60).itertuples()))
