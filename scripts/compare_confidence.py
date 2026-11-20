import os

import numpy as np
import pandas as pd

from common import CLASSES, ROOT


def main():
    output = ROOT / 'reports/confidence'
    options = {'parse_dates': ['MatchDate'], 'dtype': {'Season': str}}
    previous = pd.read_csv(ROOT / 'reports/goal_score/predictions.csv', **options)
    monthly = pd.read_csv(ROOT / 'reports/team_strength/predictions.csv', **options)
    keys = ['Season', 'MatchDate', 'HomeTeam', 'AwayTeam']
    common = previous.merge(monthly, on=keys, suffixes=('_previous', '_monthly'), validate='one_to_one')
    if len(common) != len(previous) or not common.FTR_previous.eq(common.FTR_monthly).all():
        raise ValueError('Monthly and previous predictions have different fixture coverage or labels')
    rows = []
    for name, prefix, selected_column in [('Previous goal-count', 'goal_score_p_', 'goal_score_covered'),
                                           ('Previous CatBoost + xG', 'catboost_xg_p_', 'catboost_xg_covered'),
                                           ('Monthly team strength', 'p_', 'covered')]:
        probs = common[[prefix + label for label in CLASSES]].to_numpy()
        selected = common[selected_column].to_numpy(dtype=bool)
        correct = np.array(CLASSES)[probs.argmax(axis=1)] == common.FTR_previous.to_numpy()
        count = int(selected.sum())
        rows.append({'model': name, 'eligible_matches': len(common), 'covered': count,
                     'correct': int(correct[selected].sum()), 'accuracy': float(correct[selected].mean()) if count else None,
                     'coverage': float(selected.mean())})
    result = pd.DataFrame(rows)
    result.to_csv(output / 'paired_comparison.csv', index=False)

    candidates = pd.read_csv(output / 'metrics.csv')
    candidates = candidates.loc[candidates.season.eq('pooled') & candidates.policy.eq('selected')]
    additions = candidates.loc[candidates.model.isin(['goal_xg', 'catboost_xg', 'calibrated_blend'])]
    plot = pd.concat([result, additions], ignore_index=True)
    labels = {'Previous goal-count': 'Previous goal-count',
              'Previous CatBoost + xG': 'Previous CatBoost + xG',
              'Monthly team strength': 'Monthly team strength',
              'goal_xg': 'Goal-count, broader cutoff',
              'catboost_xg': 'CatBoost, broader cutoff',
              'calibrated_blend': 'Calibrated blend'}
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in plot.itertuples():
        color = '#2e725b' if row.model.startswith('Previous') else '#3b6691'
        ax.scatter(row.coverage * 100, row.accuracy * 100, s=90, color=color, zorder=3)
        offset = (8, -24) if row.model == 'Monthly team strength' else (8, 8)
        if row.model == 'calibrated_blend':
            offset = (8, -42)
        label = f'{labels[row.model]}\n{row.correct}/{row.covered}'
        ax.annotate(label, (row.coverage * 100, row.accuracy * 100), xytext=offset,
                    textcoords='offset points', fontsize=9)
    ax.axhline(65, color='#9d4e44', linestyle='--', linewidth=1)
    ax.text(6, 65.5, '65% target', color='#9d4e44', fontsize=9)
    ax.set(xlim=(5, 43), ylim=(54, 81), xlabel='Eligible matches receiving predictions (%)',
           ylabel='Correct among issued predictions (%)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.15)
    fig.suptitle('More match predictions reduced precision', x=0.10, ha='left', fontsize=17, weight='bold')
    fig.text(0.10, 0.89, f'Common evaluation sample of {len(common)} matches. Cutoffs selected from earlier validation.', fontsize=10)
    snapshot = pd.read_csv(ROOT / 'reports/team_strength/metrics.csv')
    snapshot = snapshot.loc[snapshot.season.eq('2627_snapshot') & snapshot.policy.eq('conservative_cutoff')]
    footer = 'Retrospective follow-up on previously examined seasons.'
    if not snapshot.empty:
        record = snapshot.iloc[0]
        footer += f'\nThe monthly model scored {record.correct}/{record.covered} on the separate 2026–2027 snapshot.'
    fig.text(0.10, 0.025, footer, fontsize=9, color='#444444')
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.17)
    fig.savefig(output / 'precision_and_coverage.png', dpi=180)
    plt.close(fig)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
