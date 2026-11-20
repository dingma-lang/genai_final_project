import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import CLASSES, ROOT, write_json
from confidence_study import wilson_interval
from goal_score_study import conservative_policy, score_probabilities
from precision_study import match_rows, precision_row, read_fixtures
from train_market_confidence import load_snapshot
from train_precision65 import HOLDOUTS, SEASONS

OUTPUT = ROOT / 'reports/team_strength'
VALIDATION = ['1920', '2021', '2122', '2223', '2324']


def goal_rows(fixtures):
    home = pd.DataFrame({'attack_team': fixtures.HomeTeam, 'defence_team': fixtures.AwayTeam,
                         'at_home': 1.0})
    away = pd.DataFrame({'attack_team': fixtures.AwayTeam, 'defence_team': fixtures.HomeTeam,
                         'at_home': 0.0})
    return pd.concat([home, away], ignore_index=True)


def fit_strength(history, cutoff, alpha, half_life):
    prior = history.loc[history.MatchDate.lt(cutoff)]
    if prior.empty:
        raise ValueError('Team-strength training needs earlier fixtures')
    rows = goal_rows(prior)
    age = np.tile((cutoff - prior.MatchDate).dt.days.to_numpy(), 2)
    weights = 0.5 ** (age / half_life)
    weights /= weights.mean()
    prep = ColumnTransformer([('teams', OneHotEncoder(handle_unknown='ignore'), ['attack_team', 'defence_team']),
                              ('venue', 'passthrough', ['at_home'])])
    model = Pipeline([('features', prep), ('goals', PoissonRegressor(alpha=alpha, max_iter=1000))])
    goals = np.concatenate([prior.FTHG.to_numpy(), prior.FTAG.to_numpy()])
    return model.fit(rows, goals, goals__sample_weight=weights)


def predict_rates(model, fixtures):
    values = model.predict(goal_rows(fixtures))
    return np.column_stack([values[:len(fixtures)], values[len(fixtures):]])


def monthly_rates(fixtures, targets, alpha, half_life):
    rates = np.empty((len(targets), 2))
    for month in sorted(targets.MatchDate.dt.to_period('M').unique()):
        selected = targets.MatchDate.dt.to_period('M').eq(month).to_numpy()
        cutoff = month.start_time
        fitted = fit_strength(fixtures, cutoff, alpha, half_life)
        rates[selected] = predict_rates(fitted, targets.loc[selected])
    return rates


def main():
    sources = []
    for name in ['goalkeeper_sources', 'draw_lineup_sources', 'confidence_sources']:
        sources.extend(json.loads((ROOT / 'data' / (name + '.json')).read_text()))
    fixtures = read_fixtures(SEASONS, sources)
    eligible = match_rows(fixtures)[['Season', 'MatchDate', 'HomeTeam', 'AwayTeam', 'FTR']]
    # Goal columns are used only as training targets and evaluation labels.
    keys = ['Season', 'MatchDate', 'HomeTeam', 'AwayTeam', 'FTR']
    eligible = eligible.merge(fixtures[keys + ['FTHG', 'FTAG']], on=keys, validate='one_to_one')
    validation = eligible.loc[eligible.Season.isin(VALIDATION)].reset_index(drop=True)
    test = eligible.loc[eligible.Season.isin(HOLDOUTS)].reset_index(drop=True)
    trials, rows = [], []
    for alpha in [0.001, 0.01, 0.1]:
        for half_life in [180, 365, 730]:
            rates = monthly_rates(fixtures, validation, alpha, half_life)
            for rho in [-0.1, 0.0]:
                probs = score_probabilities(rates, rho)
                loss = float(log_loss(validation.FTR, probs, labels=CLASSES))
                settings = {'alpha': alpha, 'half_life': half_life, 'rho': rho}
                trials.append((loss, settings, probs))
                rows.append({**settings, 'validation_log_loss': loss})
            print('Validated team strengths', alpha, half_life, flush=True)
    loss, settings, probs = min(trials, key=lambda trial: trial[0])
    policy, cutoffs = conservative_policy(validation.FTR.to_numpy(), validation.Season.to_numpy(), probs)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {'settings': settings, 'validation_log_loss': loss, 'policy': policy,
                'validation_seasons': VALIDATION, 'test_seasons': HOLDOUTS,
                'updates': 'Refit on the first calendar day of each month using strictly earlier completed games; exponential time weights.',
                'inputs': 'Attack team identity, opponent defence identity, and home indicator. Earlier scores are training targets.',
                'evaluation_status': 'Retrospective follow-up after earlier project results; independent prospective evaluation remains outstanding.'}
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    pd.DataFrame(rows).to_csv(OUTPUT / 'forward_validation.csv', index=False)
    pd.DataFrame(cutoffs).to_csv(OUTPUT / 'validation_cutoffs.csv', index=False)
    print('Frozen team-strength settings', settings, 'policy', policy, flush=True)
    rates = monthly_rates(fixtures, test, settings['alpha'], settings['half_life'])
    forecasts = score_probabilities(rates, settings['rho'])
    records = []
    for season in [*HOLDOUTS, 'pooled']:
        cohort = test.Season.eq(season).to_numpy() if season != 'pooled' else np.ones(len(test), dtype=bool)
        choices = {'all_fixtures': np.ones(cohort.sum(), dtype=bool)}
        if policy:
            choices['conservative_cutoff'] = forecasts[cohort].max(axis=1) >= policy['cutoff']
        for name, called in choices.items():
            row = precision_row('monthly_team_strength', name, test.FTR.to_numpy()[cohort], forecasts[cohort], called)
            row.update({'season': season, 'eligible_matches': int(cohort.sum())})
            row['wilson_lower'], row['wilson_upper'] = wilson_interval(row['correct'], row['covered'])
            for label in CLASSES:
                actual = test.FTR.to_numpy()[cohort] == label
                predicted = np.array(CLASSES)[forecasts[cohort].argmax(axis=1)]
                row['recall_' + label] = float((called & actual & (predicted == label)).sum() / actual.sum())
            records.append(row)
    exported = test[keys].copy()
    for index, label in enumerate(CLASSES):
        exported['p_' + label] = forecasts[:, index]
    exported['covered'] = forecasts.max(axis=1) >= policy['cutoff'] if policy else False
    exported.to_csv(OUTPUT / 'predictions.csv', index=False)
    # Keep the earlier snapshot as an additional dated evaluation with frozen settings.
    _, snapshot = load_snapshot()
    all_fixtures = pd.concat([fixtures, snapshot], ignore_index=True)
    snapshot_rows = match_rows(all_fixtures)
    snapshot_rows = snapshot_rows.loc[snapshot_rows.Season.eq('2627'), keys].merge(snapshot[keys + ['FTHG', 'FTAG']], on=keys, validate='one_to_one')
    snapshot_rates = monthly_rates(all_fixtures, snapshot_rows, settings['alpha'], settings['half_life'])
    snapshot_probs = score_probabilities(snapshot_rates, settings['rho'])
    choices = {'all_fixtures': np.ones(len(snapshot_rows), dtype=bool)}
    if policy:
        choices['conservative_cutoff'] = snapshot_probs.max(axis=1) >= policy['cutoff']
    for name, called in choices.items():
        row = precision_row('monthly_team_strength', name, snapshot_rows.FTR.to_numpy(), snapshot_probs, called)
        row.update({'season': '2627_snapshot', 'eligible_matches': len(snapshot_rows)})
        row['wilson_lower'], row['wilson_upper'] = wilson_interval(row['correct'], row['covered'])
        for label in CLASSES:
            actual = snapshot_rows.FTR.to_numpy() == label
            predicted = np.array(CLASSES)[snapshot_probs.argmax(axis=1)]
            row['recall_' + label] = float((called & actual & (predicted == label)).sum() / actual.sum())
        records.append(row)
    snapshot_export = snapshot_rows[keys].copy()
    for index, label in enumerate(CLASSES):
        snapshot_export['p_' + label] = snapshot_probs[:, index]
    snapshot_export['covered'] = snapshot_probs.max(axis=1) >= policy['cutoff'] if policy else False
    snapshot_export.to_csv(OUTPUT / 'snapshot_predictions.csv', index=False)
    model_dir = ROOT / 'models/confidence'
    model_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp('2026-11-20')
    final = fit_strength(all_fixtures, cutoff, settings['alpha'], settings['half_life'])
    path = model_dir / 'team_strength.joblib'
    joblib.dump({'model': final, 'settings': settings, 'policy': policy, 'trained_before': str(cutoff.date())}, path)
    np.testing.assert_allclose(predict_rates(final, snapshot_rows), predict_rates(joblib.load(path)['model'], snapshot_rows))
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT / 'metrics.csv', index=False)
    print(result[['season', 'policy', 'covered', 'correct', 'accuracy', 'coverage', 'log_loss']].to_string(index=False))


if __name__ == '__main__':
    main()
