import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, ROOT, write_json
from model import probabilities, weekly_bootstrap
from precision_study import precision_row
from train_precision65 import HOLDOUTS
from xg_study import BASE_COLUMNS, KEYS, VALIDATION, XG_COLUMNS, build_features, fit_candidate, load_matches

OUTPUT = ROOT / 'reports/goal_score'
COLUMNS = BASE_COLUMNS + XG_COLUMNS


def fit_scores(frame, alpha, half_life_days, columns=COLUMNS):
    ages = (frame.MatchDate.max() - frame.MatchDate).dt.days.to_numpy()
    weights = np.ones(len(frame)) if half_life_days is None else 0.5 ** (ages / half_life_days)
    weights /= weights.mean()
    models = []
    for target in ['FTHG', 'FTAG']:
        model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1000))
        model.fit(frame[columns], frame[target], poissonregressor__sample_weight=weights)
        models.append(model)
    return models


def score_probabilities(rates, rho):
    rates = np.clip(np.asarray(rates), 0.05, 8.0)
    home, away = rates.T
    counts = np.arange(51)
    grid = poisson.pmf(counts[None, :], home[:, None])[:, :, None] * poisson.pmf(counts[None, :], away[:, None])[:, None, :]
    grid[:, 0, 0] *= 1 - home * away * rho
    grid[:, 0, 1] *= 1 + home * rho
    grid[:, 1, 0] *= 1 + away * rho
    grid[:, 1, 1] *= 1 - rho
    if not np.isfinite(grid).all() or (grid < 0).any():
        raise ValueError('Invalid score distribution')
    probabilities = np.column_stack([np.triu(grid, 1).sum(axis=(1, 2)),
                                     np.trace(grid, axis1=1, axis2=2),
                                     np.tril(grid, -1).sum(axis=(1, 2))])
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def predict_scores(models, frame, columns=COLUMNS):
    return np.column_stack([model.predict(frame[columns]) for model in models])


def conservative_policy(outcomes, seasons, probabilities):
    correct = np.array(CLASSES)[probabilities.argmax(axis=1)] == outcomes
    rows = []
    for cutoff in np.linspace(0.45, 0.90, 91):
        calls = probabilities.max(axis=1) >= cutoff
        if not calls.any():
            continue
        row = {'cutoff': float(cutoff), 'precision': float(correct[calls].mean()),
               'coverage': float(calls.mean()), 'calls': int(calls.sum())}
        eligible = row['precision'] >= 0.75 and row['coverage'] >= 0.10
        for season in VALIDATION[-2:]:
            cohort = seasons == season
            selected = cohort & calls
            rate = float(correct[selected].mean()) if selected.any() else 0.0
            coverage = float(selected.sum() / cohort.sum())
            row[season + '_precision'] = rate
            row[season + '_coverage'] = coverage
            eligible &= rate >= 0.70 and coverage >= 0.05
        row['eligible'] = bool(eligible)
        rows.append(row)
    eligible = [row for row in rows if row['eligible']]
    return (max(eligible, key=lambda row: (row['coverage'], row['precision'])) if eligible else None), rows


def evaluate(test, forecasts, policies):
    metrics = []
    for name, probs in forecasts.items():
        if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1):
            raise ValueError('Invalid evaluation probabilities')
        for season in [*HOLDOUTS, 'pooled']:
            cohort = test.Season.eq(season).to_numpy() if season != 'pooled' else np.ones(len(test), dtype=bool)
            selections = {'all_fixtures': np.ones(cohort.sum(), dtype=bool)}
            policy = policies.get(name)
            if policy:
                selections['conservative_cutoff'] = probs[cohort].max(axis=1) >= policy['cutoff']
            goal_policy = policies.get('goal_score')
            if goal_policy:
                same_games = forecasts['goal_score'][cohort].max(axis=1) >= goal_policy['cutoff']
                selections['same_goal_score_fixtures'] = same_games
                ranked = np.zeros(cohort.sum(), dtype=bool)
                ranked[np.argsort(-probs[cohort].max(axis=1), kind='stable')[:same_games.sum()]] = True
                selections['same_call_count_retrospective'] = ranked
            for policy_name, chosen in selections.items():
                row = precision_row(name, policy_name, test.FTR.to_numpy()[cohort], probs[cohort], chosen)
                row.update({'season': season, 'eligible_matches': int(cohort.sum())})
                for label in CLASSES:
                    actual = test.FTR.to_numpy()[cohort] == label
                    pred = np.array(CLASSES)[probs[cohort].argmax(axis=1)]
                    row['recall_' + label] = float((chosen & actual & (pred == label)).sum() / actual.sum())
                metrics.append(row)
    return pd.DataFrame(metrics)


def plot_results(metrics):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pooled = metrics.loc[metrics.season.eq('pooled')]
    comparisons = [('previous_catboost', 'all_fixtures'), ('goal_score', 'all_fixtures'),
                   ('catboost_xg', 'conservative_cutoff'), ('goal_score', 'conservative_cutoff')]
    label_options = ['Previous CatBoost\nAll eligible\nmatches', 'Goal-count model\nAll eligible\nmatches',
                     'CatBoost + xG\nSelected\nmatches', 'Goal-count model\nSelected\nmatches']
    frames, labels = [], []
    for (name, policy), label in zip(comparisons, label_options):
        found = pooled.loc[pooled.model.eq(name) & pooled.policy.eq(policy)]
        if not found.empty:
            frames.append(found)
            labels.append(label)
    rows = pd.concat(frames)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    colors = ['#718096', '#2d6a85', '#619682', '#24684c']
    for ax, column, title in zip(axes, ['accuracy', 'coverage'],
                                  ['Correct predictions (%)', 'Matches receiving predictions (%)']):
        bars = ax.bar(np.arange(len(rows)), rows[column] * 100, color=colors, width=0.65)
        ax.set_xticks(np.arange(len(rows)), labels, fontsize=9)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, loc='left', fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)
        for bar, row in zip(bars, rows.itertuples()):
            value = getattr(row, column) * 100
            label = f'{value:.1f}%'
            if column == 'accuracy':
                label += f'\n{row.correct}/{row.covered}'
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2, label, ha='center', va='bottom', fontsize=10)
    axes[0].axhline(65, color='#8d4743', linestyle='--', linewidth=1)
    axes[0].text(-0.4, 66.5, '65% target', color='#8d4743', fontsize=9)
    fig.suptitle('Higher precision with fewer match predictions', x=0.06, ha='left', fontsize=17, weight='bold')
    count = int(rows.eligible_matches.iloc[0])
    fig.text(0.06, 0.87, f'{count} eligible matches across 2024–2025 and part of 2025–2026. xG means expected goals.', fontsize=10)
    fig.text(0.06, 0.035, 'Retrospective follow-up on previously examined seasons. Cutoffs selected from earlier validation seasons.\nAn independent future evaluation is needed to measure reliability.', fontsize=9, color='#444444')
    fig.subplots_adjust(left=0.06, right=0.98, top=0.77, bottom=0.23, wspace=0.25)
    fig.savefig(OUTPUT / 'accuracy_and_coverage.png', dpi=180)
    plt.close(fig)


def main():
    fixtures, _, _ = load_matches()
    frame = build_features(fixtures).merge(fixtures[KEYS + ['FTHG', 'FTAG']], on=KEYS, validate='one_to_one')
    development = frame.loc[frame.Season.lt(HOLDOUTS[0])]
    test = frame.loc[frame.Season.isin(HOLDOUTS)].reset_index(drop=True)
    folds = [(development.loc[development.Season.lt(season)], development.loc[development.Season.eq(season)], season)
             for season in VALIDATION]
    outcomes = np.concatenate([val.FTR.to_numpy() for _, val, _ in folds])
    seasons = np.concatenate([val.Season.to_numpy() for _, val, _ in folds])
    trials, details = [], []
    for alpha in [0.1, 1.0, 10.0]:
        for half_life in [None, 730]:
            rates = []
            for train, validation, _ in folds:
                models = fit_scores(train, alpha, half_life)
                rates.append(predict_scores(models, validation))
            for rho in [-0.1, 0.0]:
                forecast = score_probabilities(np.vstack(rates), rho)
                loss = float(log_loss(outcomes, forecast, labels=CLASSES))
                config = {'alpha': alpha, 'half_life_days': half_life, 'rho': rho}
                trials.append((loss, config, forecast))
                details.append({**config, 'validation_log_loss': loss})
    loss, config, validation_probs = min(trials, key=lambda trial: trial[0])
    policies, grids = {}, []
    policy, grid = conservative_policy(outcomes, seasons, validation_probs)
    policies['goal_score'] = policy
    grids.extend({'model': 'goal_score', **row} for row in grid)
    # Apply the identical conservative rule to the already selected classifier.
    manifest = json.loads((ROOT / 'reports/xg/frozen_policy.json').read_text())
    cat_config = manifest['models']['catboost_xg']
    cat_validation = []
    for train, validation, _ in folds:
        model = fit_candidate(train, cat_config['family'], cat_config['columns'], cat_config['settings'])
        cat_validation.append(probabilities(model, validation[cat_config['columns']]))
    policy, grid = conservative_policy(outcomes, seasons, np.vstack(cat_validation))
    policies['catboost_xg'] = policy
    grids.extend({'model': 'catboost_xg', **row} for row in grid)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = {'configuration': config, 'validation_log_loss': loss, 'policies': policies,
              'features': COLUMNS, 'validation_seasons': VALIDATION, 'test_seasons': HOLDOUTS,
              'policy_requirements': '75% validation hit rate and 10% coverage overall; 70% hit rate and 5% coverage in each of the last two validation seasons.',
              'evaluation_status': 'Exploratory second experiment after inspecting the xG follow-up. Same previously examined seasons; independent future evaluation still needed.',
              'score_model': 'Separate regularized Poisson regressions for home and away goals; optional Dixon-Coles low-score probability adjustment selected by validation log loss.',
              'rate_bounds': [0.05, 8.0], 'score_grid': [0, 50]}
    write_json(OUTPUT / 'frozen_policy.json', frozen)
    pd.DataFrame(details).to_csv(OUTPUT / 'forward_validation.csv', index=False)
    pd.DataFrame(grids).to_csv(OUTPUT / 'validation_cutoffs.csv', index=False)
    models = fit_scores(development, config['alpha'], config['half_life_days'])
    path = ROOT / 'models/xg/goal_score.joblib'
    joblib.dump({'models': models, 'rho': config['rho'], 'columns': COLUMNS}, path)
    rates = predict_scores(models, test)
    forecast = score_probabilities(rates, config['rho'])
    restored = joblib.load(path)
    np.testing.assert_allclose(score_probabilities(predict_scores(restored['models'], test), restored['rho']), forecast)
    xg_predictions = pd.read_csv(ROOT / 'reports/xg/predictions.csv', parse_dates=['MatchDate'], dtype={'Season': str})
    paired = test[KEYS].merge(xg_predictions, on=KEYS, how='left', validate='one_to_one')
    forecasts = {'goal_score': forecast, 'catboost_xg': paired[['catboost_xg_p_' + label for label in CLASSES]].to_numpy()}
    for name in ['previous_catboost', 'previous_elo_logistic', 'previous_market_benchmark']:
        forecasts[name] = paired[[name + '_p_' + label for label in CLASSES]].to_numpy()
    exported = test[KEYS + ['FTR']].copy()
    exported['expected_home_goals'] = np.clip(rates[:, 0], 0.05, 8.0)
    exported['expected_away_goals'] = np.clip(rates[:, 1], 0.05, 8.0)
    for name, probs in forecasts.items():
        for index, label in enumerate(CLASSES):
            exported[name + '_p_' + label] = probs[:, index]
        if policies.get(name):
            exported[name + '_covered'] = probs.max(axis=1) >= policies[name]['cutoff']
    exported.to_csv(OUTPUT / 'predictions.csv', index=False)
    result = evaluate(test, forecasts, policies)
    result.to_csv(OUTPUT / 'metrics.csv', index=False)
    plot_results(result)
    comparison = weekly_bootstrap(test, forecasts['previous_catboost'], forecasts['goal_score'])
    comparison['limitation'] = 'Exploratory weekly block interval across two previously examined seasons; model selection and all team dependence are not captured.'
    frozen['goal_score_minus_previous_catboost'] = comparison
    write_json(OUTPUT / 'frozen_policy.json', frozen)
    print('Selected settings', config, 'validation loss', loss, flush=True)
    print('Conservative policies', policies, flush=True)
    print(result.loc[result.season.eq('pooled'), ['model', 'policy', 'covered', 'correct', 'accuracy', 'coverage', 'log_loss', 'called_D']].to_string(index=False))


if __name__ == '__main__':
    main()
