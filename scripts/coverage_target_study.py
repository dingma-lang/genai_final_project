import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from common import CLASSES, ROOT, write_json
from confidence_study import wilson_interval
from goal_score_study import score_probabilities
from precision_study import precision_row
from shared_goal_study import fit_model, predict_rates
from train_precision65 import HOLDOUTS
from xg_study import KEYS, VALIDATION, build_features, load_matches

OUTPUT = ROOT / 'reports/coverage_target'
MODEL_DIR = ROOT / 'models/coverage_target'
TARGET_CALLS = 100
REFERENCE_MATCHES = 619
TARGET_COVERAGE = TARGET_CALLS / REFERENCE_MATCHES
HISTORY_WINDOW = 350


def rank_selection(confidence, count):
    confidence = np.asarray(confidence, dtype=float)
    if confidence.ndim != 1 or not np.isfinite(confidence).all() or not 0 <= count <= len(confidence):
        raise ValueError('Invalid confidence values or prediction count')
    selected = np.zeros(len(confidence), dtype=bool)
    selected[np.argsort(-confidence, kind='stable')[:count]] = True
    return selected


def rolling_selection(dates, confidence, initial_history, coverage=TARGET_COVERAGE, window=HISTORY_WINDOW):
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    confidence = np.asarray(confidence, dtype=float)
    history = list(np.asarray(initial_history, dtype=float))
    if len(dates) != len(confidence) or not dates.is_monotonic_increasing or not history:
        raise ValueError('Expected chronological fixtures and earlier confidence history')
    if not 0 < coverage < 1 or window < 1 or not np.isfinite([*history, *confidence]).all():
        raise ValueError('Invalid rolling confidence policy')
    selected = np.zeros(len(confidence), dtype=bool)
    cutoffs = np.empty(len(confidence))
    for _, ids in dates.groupby(dates.dt.normalize(), sort=True).groups.items():
        indices = np.asarray(ids)
        cutoff = float(np.quantile(history[-window:], 1 - coverage, method='higher'))
        cutoffs[indices] = cutoff
        selected[indices] = confidence[indices] >= cutoff
        history.extend(confidence[indices].tolist())
    return selected, cutoffs


def add_ensembles(forecasts):
    output = dict(forecasts)
    output['goals_ensemble'] = (forecasts['linear_goals'] + forecasts['catboost_goals']) / 2
    output['all_models_ensemble'] = np.mean(list(forecasts.values()), axis=0)
    return output


def predict_bundle(bundle, frame):
    parts = []
    for member in bundle['members'].values():
        rates = predict_rates(member['model'], member['family'], frame)
        parts.append(score_probabilities(rates, 0))
    return np.mean(parts, axis=0)


def metric_rows(test, name, forecast, policies):
    rows = []
    predicted = np.array(CLASSES)[forecast.argmax(axis=1)]
    for season in [*HOLDOUTS, 'pooled']:
        cohort = test.Season.eq(season).to_numpy() if season != 'pooled' else np.ones(len(test), dtype=bool)
        for policy_name, selected in policies.items():
            row = precision_row(name, policy_name, test.FTR.to_numpy()[cohort], forecast[cohort], selected[cohort])
            row.update({'season': season, 'eligible_matches': int(cohort.sum())})
            row['wilson_lower'], row['wilson_upper'] = wilson_interval(row['correct'], row['covered'])
            row['meets_requested_target'] = bool(90 <= row['covered'] <= 110 and row['accuracy'] >= 0.70) if row['covered'] else False
            for label in CLASSES:
                actual = test.FTR.to_numpy() == label
                denominator = int((actual & cohort).sum())
                row['recall_' + label] = float((actual & cohort & selected & (predicted == label)).sum() / denominator) if denominator else None
            rows.append(row)
    return rows


def main():
    fixtures, _, _ = load_matches()
    frame = build_features(fixtures).merge(fixtures[KEYS + ['FTHG', 'FTAG', 'home_xg', 'away_xg']], on=KEYS, validate='one_to_one')
    development = frame.loc[frame.Season.lt(HOLDOUTS[0])].reset_index(drop=True)
    training = development.dropna(subset=['home_xg', 'away_xg'])
    test = frame.loc[frame.Season.isin(HOLDOUTS)].reset_index(drop=True)
    if len(test) != REFERENCE_MATCHES:
        raise ValueError('The reference sample changed; define the desired coverage before evaluation')
    prior = json.loads((ROOT / 'reports/shared_goals/frozen_policy.json').read_text())
    validation_frames, parts = [], {name: [] for name in prior['models']}
    for season in VALIDATION:
        train = training.loc[training.Season.lt(season)]
        validation = development.loc[development.Season.eq(season)]
        if train.empty or validation.empty or train.MatchDate.max() >= validation.MatchDate.min():
            raise ValueError('Invalid validation chronology')
        validation_frames.append(validation)
        for name, config in prior['models'].items():
            fitted = fit_model(train, config['family'], config['target'], config['config'])
            rates = predict_rates(fitted, config['family'], validation)
            parts[name].append(score_probabilities(rates, 0))
        print('Completed validation forecasts', season, flush=True)
    validation = pd.concat(validation_frames).reset_index(drop=True)
    forecasts = add_ensembles({name: np.vstack(values) for name, values in parts.items()})
    summaries, seasonal, settings = [], [], {}
    for name, forecast in forecasts.items():
        selected = np.zeros(len(validation), dtype=bool)
        correct = np.array(CLASSES)[forecast.argmax(axis=1)] == validation.FTR.to_numpy()
        recent_precision = []
        for season in VALIDATION:
            indices = np.flatnonzero(validation.Season.eq(season).to_numpy())
            count = int(round(len(indices) * TARGET_COVERAGE))
            picked = rank_selection(forecast[indices].max(axis=1), count)
            selected[indices] = picked
            precision = float(correct[indices][picked].mean())
            seasonal.append({'model': name, 'season': season, 'calls': count,
                             'correct': int(correct[indices][picked].sum()), 'precision': precision})
            if season in VALIDATION[-2:]:
                recent_precision.append(precision)
        precision = float(correct[selected].mean())
        loss = float(log_loss(validation.FTR, forecast, labels=CLASSES))
        confidence = forecast.max(axis=1)
        cutoff = float(np.quantile(confidence, 1 - TARGET_COVERAGE, method='higher'))
        settings[name] = {'validation_precision': precision, 'validation_log_loss': loss,
                          'minimum_recent_precision': min(recent_precision), 'fixed_cutoff': cutoff,
                          'initial_confidence_history': confidence[-HISTORY_WINDOW:].tolist(),
                          'eligible': precision >= 0.70 and min(recent_precision) >= 0.70}
        summaries.append({'model': name, 'calls': int(selected.sum()), 'correct': int(correct[selected].sum()),
                          **{key: value for key, value in settings[name].items() if key != 'initial_confidence_history'}})
    eligible = [name for name, config in settings.items() if config['eligible']]
    winner = max(eligible, key=lambda name: (settings[name]['validation_precision'],
                                            settings[name]['minimum_recent_precision'],
                                            -settings[name]['validation_log_loss'])) if eligible else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {'requested_precision': 0.70, 'requested_calls': TARGET_CALLS, 'about_100_range': [90, 110],
                'reference_matches': REFERENCE_MATCHES, 'target_coverage': TARGET_COVERAGE,
                'validation_seasons': VALIDATION, 'test_seasons': HOLDOUTS,
                'selected_by_validation_precision': winner,
                'selection': 'Use about 16.2% of each validation season. Require 70% pooled precision and 70% in each of the two latest validation seasons. Select greatest pooled precision, then worst recent-season precision, then lowest log loss.',
                'batch_policy': 'Rank a complete candidate pool by confidence and select 100 using stable, outcome-independent tie breaking. Retrospective batch comparison.',
                'fixed_policy': 'Cutoff is the earlier validation confidence quantile for the target coverage.',
                'rolling_policy': 'Use the corresponding quantile of the previous 350 confidence scores, initialized from earlier validation. Update history after all matches on each calendar date have been selected.',
                'evaluation_status': 'Retrospective follow-up on previously examined seasons; model settings and sample size were already known. Independent future validation remains outstanding.',
                'models': {name: {key: value for key, value in config.items() if key != 'initial_confidence_history'} for name, config in settings.items()}}
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    pd.DataFrame(summaries).to_csv(OUTPUT / 'validation_summary.csv', index=False)
    pd.DataFrame(seasonal).to_csv(OUTPUT / 'validation_by_season.csv', index=False)
    print('Frozen validation selection', winner, flush=True)
    old = pd.read_csv(ROOT / 'reports/shared_goals/predictions.csv', parse_dates=['MatchDate'], dtype={'Season': str})
    old = test[KEYS + ['FTR']].merge(old, on=KEYS, how='left', validate='one_to_one', suffixes=('', '_old'))
    if not old.FTR.eq(old.FTR_old).all():
        raise ValueError('Reference prediction outcomes disagree')
    test_forecasts = add_ensembles({name: old[[name + '_p_' + label for label in CLASSES]].to_numpy() for name in prior['models']})
    rows = []
    predictions = test[KEYS + ['FTR']].copy()
    for name, forecast in test_forecasts.items():
        if not np.isfinite(forecast).all() or not np.allclose(forecast.sum(axis=1), 1):
            raise ValueError('Invalid saved model probabilities')
        confidence = forecast.max(axis=1)
        rolling, cutoffs = rolling_selection(test.MatchDate, confidence, settings[name]['initial_confidence_history'])
        policies = {'top_100_batch': rank_selection(confidence, TARGET_CALLS),
                    'fixed_cutoff': confidence >= settings[name]['fixed_cutoff'], 'rolling_cutoff': rolling}
        earlier_policy = prior['models'].get(name, {}).get('conservative_policy')
        if earlier_policy:
            policies['previous_conservative_cutoff'] = confidence >= earlier_policy['cutoff']
        rows.extend(metric_rows(test, name, forecast, policies))
        for index, label in enumerate(CLASSES):
            predictions[name + '_p_' + label] = forecast[:, index]
        for policy_name, selected in policies.items():
            predictions[name + '_' + policy_name] = selected
        predictions[name + '_rolling_threshold'] = cutoffs
    benchmark_file = pd.read_csv(ROOT / 'reports/goal_score/predictions.csv', parse_dates=['MatchDate'], dtype={'Season': str})
    benchmarks = test[KEYS].merge(benchmark_file, on=KEYS, how='left', validate='one_to_one')
    for name in ['previous_catboost', 'previous_market_benchmark']:
        forecast = benchmarks[[name + '_p_' + label for label in CLASSES]].to_numpy()
        if not np.isfinite(forecast).all():
            raise ValueError('Missing benchmark probabilities')
        policies = {'top_100_batch': rank_selection(forecast.max(axis=1), TARGET_CALLS)}
        rows.extend(metric_rows(test, name, forecast, policies))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / 'metrics.csv', index=False)
    predictions.to_csv(OUTPUT / 'predictions.csv', index=False)
    if winner:
        members = [winner]
        if winner == 'goals_ensemble':
            members = ['linear_goals', 'catboost_goals']
        elif winner == 'all_models_ensemble':
            members = list(prior['models'])
        bundle = {'members': {name: joblib.load(ROOT / 'models/shared_goals' / (name + '.joblib')) for name in members},
                  'selected_model': winner, 'fixed_cutoff': settings[winner]['fixed_cutoff'],
                  'target_coverage': TARGET_COVERAGE, 'history_window': HISTORY_WINDOW,
                  'initial_confidence_history': settings[winner]['initial_confidence_history'],
                  'previous_conservative_policy': prior['models'].get(winner, {}).get('conservative_policy')}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / 'selected_forecaster.joblib'
        joblib.dump(bundle, path)
        np.testing.assert_allclose(predict_bundle(joblib.load(path), test), test_forecasts[winner], rtol=1e-10, atol=1e-12)
        selected = predictions[winner + '_top_100_batch']
        calls = predictions.loc[selected, KEYS + ['FTR']].copy()
        forecast = test_forecasts[winner][selected]
        calls['prediction'] = np.array(CLASSES)[forecast.argmax(axis=1)]
        calls['confidence'] = forecast.max(axis=1)
        calls['correct'] = calls.prediction.eq(calls.FTR)
        calls.to_csv(OUTPUT / 'selected_100_matches.csv', index=False)
        earlier_policy = bundle['previous_conservative_policy']
        manifest['selected_model_previous_conservative_policy'] = earlier_policy
        write_json(OUTPUT / 'frozen_policy.json', manifest)
    print(pd.DataFrame(summaries)[['model', 'calls', 'correct', 'validation_precision', 'eligible']].to_string(index=False))
    print(metrics.loc[metrics.season.eq('pooled'), ['model', 'policy', 'covered', 'correct', 'accuracy', 'coverage', 'meets_requested_target']].to_string(index=False))


if __name__ == '__main__':
    main()
