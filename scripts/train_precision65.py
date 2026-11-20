import json

import catboost
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

from common import CLASSES, ROOT, write_json
from model import pipeline, probabilities
from precision_study import FEATURE_SETS, ODDS, SEASON_CODES, match_rows, precision_row, read_fixtures

SEASONS = [*SEASON_CODES, '1920', '2021', '2122', '2223', '2324', '2425', '2526']
HOLDOUTS = ['2425', '2526']
VALIDATION = [code for code in SEASONS if '1617' <= code < HOLDOUTS[0]]
COLUMNS = FEATURE_SETS['all_signals']
TARGET = 0.65
DEVELOPMENT_FLOOR = 0.69
MIN_COVERAGE = 0.15
THRESHOLDS = np.linspace(0.35, 0.90, 111)
MODELS = {
    'elo_logistic': [{'C': c} for c in [0.01, 0.1, 1.0, 10.0]],
    'catboost': [{'depth': depth, 'iterations': count} for depth in [3, 4, 5] for count in [200, 400]],
}


def fit_model(name, params, train):
    if name == 'elo_logistic':
        return pipeline(COLUMNS, 'logistic', params).fit(train[COLUMNS], train.FTR)
    return CatBoostClassifier(**params, loss_function='MultiClass', learning_rate=0.03, l2_leaf_reg=5,
                              random_seed=244, thread_count=2, allow_writing_files=False,
                              verbose=False).fit(train[COLUMNS], train.FTR)


def market_probability(data):
    inverse = 1 / data[ODDS].to_numpy(dtype=float)
    return inverse / inverse.sum(axis=1, keepdims=True)


def validation_forecasts(development):
    folds = []
    for season in VALIDATION:
        fit = development.loc[development.Season.lt(season)]
        validation = development.loc[development.Season.eq(season)]
        if fit.empty or validation.empty or fit.MatchDate.max() >= validation.MatchDate.min():
            raise ValueError(f'Invalid dates in validation fold {season}')
        folds.append((fit, validation, season))
    outcomes = pd.concat([val.FTR for _, val, _ in folds]).to_numpy()
    forecasts = {'market_benchmark': np.vstack([market_probability(val) for _, val, _ in folds])}
    selected = {}
    validation_detail = []
    for name, configs in MODELS.items():
        scores = []
        for params in configs:
            by_fold = []
            for fit, validation, season in folds:
                fitted = fit_model(name, params, fit)
                probs = probabilities(fitted, validation[COLUMNS])
                by_fold.append(probs)
                validation_detail.append({'model': name, 'configuration': json.dumps(params, sort_keys=True),
                                          'validation_season': season, 'fitted_rows': len(fit), 'validation_rows': len(validation),
                                          'log_loss': log_loss(validation.FTR, probs, labels=CLASSES)})
            pooled = np.vstack(by_fold)
            score = float(log_loss(outcomes, pooled, labels=CLASSES))
            scores.append((score, params, pooled))
        score, params, forecast = min(scores, key=lambda row: row[0])
        selected[name] = {'parameters': params, 'selection_log_loss': score}
        forecasts[name] = forecast
        print(f'{name} selected {params}, forward validation log loss {score:.4f}', flush=True)
    return outcomes, forecasts, selected, pd.DataFrame(validation_detail)


def choose_cutoff(outcomes, forecast):
    labels = np.array(CLASSES)[forecast.argmax(axis=1)]
    correct = labels == outcomes
    confidence = forecast.max(axis=1)
    rows = []
    for cutoff in THRESHOLDS:
        selected = confidence >= cutoff
        count = int(selected.sum())
        if count:
            precision = float(correct[selected].mean())
            coverage = float(selected.mean())
            rows.append({'cutoff': float(cutoff), 'calls': count, 'precision': precision, 'coverage': coverage,
                         'eligible': precision >= DEVELOPMENT_FLOOR and coverage >= MIN_COVERAGE})
    feasible = [row for row in rows if row['eligible']]
    policy = max(feasible, key=lambda row: (row['calls'], row['precision'], -row['cutoff'])) if feasible else None
    return policy, rows


def measure(name, cohort, outcomes, probs, policy, selected):
    row = precision_row(name, policy, outcomes, probs, selected)
    row['season'] = cohort
    row['eligible_matches'] = len(outcomes)
    row['macro_precision_zero_for_no_calls'] = float(np.mean([row['precision_' + label] or 0 for label in CLASSES]))
    labels = np.array(CLASSES)[probs.argmax(axis=1)]
    for label in CLASSES:
        # Abstained matches count as missed opportunities when measuring recall.
        actual = outcomes == label
        row['recall_' + label] = float((selected & (labels == label) & actual).sum() / actual.sum()) if actual.any() else None
    row['hit_target'] = bool(row['covered'] and row['accuracy'] >= TARGET)
    return row


def score_holdouts(holdout, forecasts, cutoffs):
    fixtures = holdout[['Season', 'MatchDate', 'HomeTeam', 'AwayTeam', 'FTR']].copy()
    rows = []
    for name, forecast in forecasts.items():
        policy = cutoffs[name]
        selection = forecast.max(axis=1) >= policy['cutoff'] if policy else np.zeros(len(holdout), dtype=bool)
        for idx, outcome in enumerate(CLASSES):
            fixtures[f'{name}_p_{outcome}'] = forecast[:, idx]
        fixtures[f'{name}_covered'] = selection
        for cohort in [*HOLDOUTS, 'pooled']:
            season = holdout.Season.eq(cohort).to_numpy() if cohort != 'pooled' else np.ones(len(holdout), dtype=bool)
            outcome = holdout.FTR.to_numpy()[season]
            rows.append(measure(name, cohort, outcome, forecast[season], 'all_fixtures',
                                np.ones(int(season.sum()), dtype=bool)))
            if policy is not None:
                rows.append(measure(name, cohort, outcome, forecast[season], 'confidence_cutoff', selection[season]))
    return pd.DataFrame(rows), fixtures


def main():
    sources = []
    for name in ['goalkeeper_sources', 'draw_lineup_sources', 'confidence_sources']:
        sources.extend(json.loads((ROOT / 'data' / (name + '.json')).read_text()))
    fixtures = read_fixtures(SEASONS, sources)
    feature_rows = match_rows(fixtures)
    development = feature_rows.loc[feature_rows.Season.lt(HOLDOUTS[0])]
    holdout = feature_rows.loc[feature_rows.Season.isin(HOLDOUTS)].reset_index(drop=True)
    if development.MatchDate.max() >= holdout.MatchDate.min() or set(holdout.Season) != set(HOLDOUTS):
        raise ValueError('Improper holdout chronology')

    outcomes, validation, selected, validation_details = validation_forecasts(development)
    cutoffs = {}
    grid = []
    for name, forecast in validation.items():
        policy, candidates = choose_cutoff(outcomes, forecast)
        cutoffs[name] = policy
        grid.extend({'model': name, **row} for row in candidates)
    feasible = [name for name in MODELS if cutoffs[name] is not None]
    champion = max(feasible, key=lambda name: (cutoffs[name]['coverage'], -selected[name]['selection_log_loss'])) if feasible else None

    report_dir = ROOT / 'reports/precision65'
    report_dir.mkdir(parents=True, exist_ok=True)
    models_dir = ROOT / 'models/precision65'
    models_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / 'frozen_policy.json', {
        'forecast_time': 'Football features use calendar dates strictly earlier than the target fixture.',
        'market_benchmark_timing': 'Archived pre-match Bet365 quotes have no per-fixture quote timestamp here.',
        'target': TARGET, 'development_precision_floor': DEVELOPMENT_FLOOR, 'min_development_coverage': MIN_COVERAGE,
        'development_matches': len(development), 'validation_matches': len(outcomes),
        'development_through': '2324', 'validation_seasons': VALIDATION, 'holdout_seasons': HOLDOUTS,
        'features': COLUMNS, 'catboost_version': catboost.__version__,
        'model_settings': selected, 'cutoffs': cutoffs, 'trained_model_selected_by_development_coverage': champion,
        'policy_order': 'Choose model settings by expanding validation log loss, then maximize validation coverage among cutoffs meeting the floor. Choose the trained model by validation coverage.',
    })
    validation_details.to_csv(report_dir / 'forward_validation.csv', index=False)
    pd.DataFrame(grid).to_csv(report_dir / 'validation_cutoffs.csv', index=False)
    print(f'Cutoffs frozen. Trained model selected in development: {champion}.', flush=True)

    forecasts = {}
    for name, details in selected.items():
        fitted = fit_model(name, details['parameters'], development)
        if name == 'catboost':
            fitted.save_model(models_dir / 'catboost.cbm')
        else:
            joblib.dump(fitted, models_dir / 'elo_logistic.joblib')
        forecasts[name] = probabilities(fitted, holdout[COLUMNS])
    forecasts['market_benchmark'] = market_probability(holdout)
    metrics, predictions = score_holdouts(holdout, forecasts, cutoffs)
    metrics.to_csv(report_dir / 'metrics.csv', index=False)
    predictions.to_csv(report_dir / 'predictions.csv', index=False)
    print(metrics[['model', 'policy', 'season', 'covered', 'correct', 'accuracy',
                   'coverage', 'log_loss', 'called_D', 'hit_target']].to_string(index=False))


if __name__ == '__main__':
    main()
