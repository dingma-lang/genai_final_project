"""Score the leakage probe: how often the model identifies each held-out match and recalls
its result at every masking level.

Level L0 is the existing names-and-date diagnostic (data/codex_metadata_diagnostic.jsonl),
where the teams and date are given, so only the result is scored.

Usage (from the repository root):
    python scripts/score_leakage.py

Reads   reports/leakage/answers_L*.jsonl, data/processed/previews_with_results.csv,
        reports/baseline/test_predictions.csv, data/codex_metadata_diagnostic.jsonl (optional)
Writes  reports/leakage/summary.csv      one row per level, with 95% Wilson intervals
        reports/leakage/per_match.csv    one row per match and level: truth, guesses, correct flags
        reports/leakage/leakage_by_level.png
"""
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE, ROOT, TEST_SEASON, load_analysis  # noqa: E402
from masking_lists import CLUB_ALIASES, SHARED_NICKNAMES  # noqa: E402

OUT = ROOT / 'reports/leakage'
LEVEL_LABELS = {
    'L0': 'L0 team names + date',
    'L1': 'L1 full preview',
    'L2': 'L2 teams masked',
    'L3': 'L3 + byline, dates',
    'L4': 'L4 + people, clubs, venues',
    'L5': 'L5 + time cues',
}


def norm(name):
    name = str(name).lower().replace('&', 'and').replace('.', '')
    # Formal names: "Arsenal Football Club", "Hull City Association Football Club", "Arsenal F.C."
    name = re.sub(r'\b(association football club|football club|fc|afc|the)\b', ' ', name)
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', name).split())


ALIASES = {}
for club, names in CLUB_ALIASES.items():
    for n in names + [club]:
        if n not in SHARED_NICKNAMES:
            ALIASES[norm(n)] = club
ALIASES.update({norm(k): v for k, v in {'Man Utd': 'Man United', 'Manchester Utd': 'Man United',
                                         'Wolverhampton': 'Wolves', 'Brighton and Hove': 'Brighton'}.items()})


def club_of(guess):
    """Map a free-text club guess to the dataset's club name, or None."""
    return ALIASES.get(norm(guess))


def season_of(guess):
    m = re.search(r'(20\d{2})\s*[-–/]\s*(?:20)?(\d{2})\b', str(guess))
    if m:
        return f'{m.group(1)}-{m.group(2)}'
    m = re.fullmatch(r'\s*(\d{2})\s*[-–/]\s*(\d{2})\s*', str(guess))
    return f'20{m.group(1)}-{m.group(2)}' if m else None


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def main():
    df = load_analysis()
    test = df[df.Season.eq(TEST_SEASON) & df[BASE].notna().all(axis=1)].set_index('MatchID')
    base = pd.read_csv(ROOT / 'reports/baseline/test_predictions.csv').set_index('MatchID')
    baseline_acc = float((base.loc[test.index, 'logistic__baseline_prediction'] == test.FTR).mean())
    home_acc = float((test.FTR == 'H').mean())

    rows, per = [], []
    meta_path = ROOT / 'data/codex_metadata_diagnostic.jsonl'
    if meta_path.exists():
        recs = [json.loads(line) for line in meta_path.read_text().splitlines()]
        correct = [r['result'] == test.loc[r['MatchID'], 'FTR'] for r in recs if r['MatchID'] in test.index]
        k, n = sum(correct), len(correct)
        lo, hi = wilson(k, n)
        rows.append({'level': 'L0', 'n': n, 'both_teams_identified': np.nan, 'season_identified': np.nan,
                     'exact_match_identified': np.nan, 'result_correct': k / n, 'result_ci_low': lo,
                     'result_ci_high': hi, 'log_loss': np.nan})

    for path in sorted(OUT.glob('answers_L*.jsonl')):
        level = path.stem.split('_')[1]
        recs = [json.loads(line) for line in path.read_text().splitlines()]
        flags = []
        for r in recs:
            t = test.loc[r['MatchID']]
            home_ok = club_of(r['home']) == t.HomeStd
            away_ok = club_of(r['away']) == t.AwayStd
            season_ok = season_of(r['season']) == t.Season
            p_true = max(r[{'H': 'p_home', 'D': 'p_draw', 'A': 'p_away'}[t.FTR]], 1e-15)
            flags.append(dict(home_ok=home_ok, away_ok=away_ok, both=home_ok and away_ok, season_ok=season_ok,
                              exact=home_ok and away_ok and season_ok, result_ok=r['result'] == t.FTR,
                              nll=-math.log(p_true)))
            per.append({'level': level, 'MatchID': r['MatchID'], 'true_home': t.HomeStd, 'true_away': t.AwayStd,
                        'true_season': t.Season, 'true_result': t.FTR, 'guess_home': r['home'],
                        'guess_away': r['away'], 'guess_season': r['season'], 'id_confidence': r['id_confidence'],
                        'guess_result': r['result'], 'p_home': r['p_home'], 'p_draw': r['p_draw'],
                        'p_away': r['p_away'], **flags[-1]})
        f = pd.DataFrame(flags)
        n, k = len(f), int(f.result_ok.sum())
        lo, hi = wilson(k, n)
        b_lo, b_hi = wilson(int(f.both.sum()), n)
        rows.append({'level': level, 'n': n, 'both_teams_identified': f.both.mean(), 'both_ci_low': b_lo,
                     'both_ci_high': b_hi, 'season_identified': f.season_ok.mean(),
                     'exact_match_identified': f.exact.mean(), 'result_correct': k / n,
                     'result_ci_low': lo, 'result_ci_high': hi, 'log_loss': f.nll.mean(),
                     'result_correct_when_identified': f.result_ok[f.exact].mean() if f.exact.any() else np.nan,
                     'result_correct_when_not_identified': f.result_ok[~f.exact].mean() if (~f.exact).any() else np.nan})
    if not rows:
        raise SystemExit('No answers found in reports/leakage/. Run scripts/leakage_probe.py first.')

    summary = pd.DataFrame(rows).sort_values('level')
    summary['baseline_model_accuracy'] = baseline_acc
    summary['always_home_accuracy'] = home_acc
    summary.to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(per).to_csv(OUT / 'per_match.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary))
    ax.plot(x, summary.result_correct * 100, marker='o', color='#c0392b', label='Result recalled correctly')
    ax.fill_between(x, summary.result_ci_low * 100, summary.result_ci_high * 100, color='#c0392b', alpha=0.12)
    ident = summary.both_teams_identified * 100
    ax.plot(x[ident.notna()], ident[ident.notna()], marker='s', color='#2c3e50', label='Both teams identified')
    ax.axhline(baseline_acc * 100, ls='--', color='#7f8c8d', lw=1)
    ax.text(x[0], baseline_acc * 100 + 1.5, f'Recent-results model {baseline_acc:.0%}', ha='left', fontsize=8, color='#555')
    ax.axhline(home_acc * 100, ls=':', color='#7f8c8d', lw=1)
    ax.text(x[0], home_acc * 100 - 4, f'Always home {home_acc:.0%}', ha='left', fontsize=8, color='#555')
    ax.set_xticks(x, [LEVEL_LABELS.get(l, l) for l in summary.level], rotation=20, ha='right', fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel('% of 279 held-out matches')
    ax.set_title('Match identification and result recall by masking level')
    ax.legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0, 0.08))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'leakage_by_level.png', dpi=180)
    cols = ['level', 'n', 'both_teams_identified', 'season_identified', 'exact_match_identified', 'result_correct', 'log_loss']
    print(summary[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    guesses = pd.concat([pd.DataFrame(per)[c] for c in ('guess_home', 'guess_away')]) if per else pd.Series(dtype=str)
    unmapped = sorted({g for g in guesses if norm(g) not in ('unknown', '') and club_of(g) is None})
    if unmapped:
        print(f'\nTeam guesses that matched no club in the sample ({len(unmapped)}); add any misspelled real club '
              f'to ALIASES in this script and rerun:\n  ' + '; '.join(unmapped[:80]))
    print(f'Recent-results model on the same matches: {baseline_acc:.3f}; always home: {home_acc:.3f}')


if __name__ == '__main__':
    main()
