import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

from common import CLASSES, ROOT, write_json
from coverage_target_study import rank_selection, rolling_selection
from goal_score_study import COLUMNS
from model import pipeline, probabilities
from opponent_form_study import COVERAGE, FORM_SIGNALS, REFERENCE_MATCHES, TARGET_CALLS, evaluate, opponent_form
from precision_study import read_fixtures
from train_market_confidence import CLOSING, ODDS_COLUMNS, market_probability
from train_precision65 import HOLDOUTS, SEASONS
from xg_study import BASE_COLUMNS, KEYS, build_features, load_matches

OUTPUT = ROOT / 'reports/precision150/market'
VALIDATION = ['2122', '2223', '2324']
FEATURES = {'odds_only': ODDS_COLUMNS, 'odds_history': [*ODDS_COLUMNS, *BASE_COLUMNS],
            'odds_xg': [*ODDS_COLUMNS, *COLUMNS],
            'odds_opponent_form': [*ODDS_COLUMNS, *COLUMNS, *[side + '_' + signal for side in ['home', 'away'] for signal in FORM_SIGNALS]]}


def fit_model(frame, config):
    columns = FEATURES[config['features']]
    age = (frame.MatchDate.max() - frame.MatchDate).dt.days.to_numpy()
    weights = np.ones(len(frame)) if config['half_life'] is None else 0.5 ** (age / config['half_life'])
    weights /= weights.mean()
    if config['family'] == 'logistic':
        fitted = pipeline(columns, 'logistic', {'C': config['C']})
        return fitted.fit(frame, frame.FTR, model__sample_weight=weights)
    fitted = CatBoostClassifier(depth=config['depth'], iterations=300, learning_rate=0.03,
                                loss_function='MultiClass', l2_leaf_reg=10, random_seed=244,
                                thread_count=2, allow_writing_files=False, verbose=False)
    return fitted.fit(frame[columns], frame.FTR, sample_weight=weights)


def configurations():
    for features in FEATURES:
        for penalty in [0.01, 0.1, 1.0]:
            for half_life in [None, 730]:
                yield {'features': features, 'family': 'logistic', 'C': penalty, 'half_life': half_life}
        for depth in [2, 3]:
            yield {'features': features, 'family': 'catboost', 'depth': depth, 'half_life': None}


def validation_score(frame, forecast):
    correct = np.array(CLASSES)[forecast.argmax(axis=1)] == frame.FTR.to_numpy()
    selected = np.zeros(len(frame), dtype=bool)
    recent = []
    for season in VALIDATION:
        indices = np.flatnonzero(frame.Season.eq(season).to_numpy())
        picked = rank_selection(forecast[indices].max(axis=1), int(round(len(indices) * COVERAGE)))
        selected[indices] = picked
        recent.append(float(correct[indices][picked].mean()))
    return {'precision': float(correct[selected].mean()), 'calls': int(selected.sum()),
            'correct': int(correct[selected].sum()), 'minimum_season_precision': min(recent),
            'log_loss': float(log_loss(frame.FTR, forecast, labels=CLASSES))}


def main():
    fixtures, _, _ = load_matches()
    sources = []
    for name in ['goalkeeper_sources', 'draw_lineup_sources', 'confidence_sources']:
        sources.extend(json.loads((ROOT / 'data' / (name + '.json')).read_text()))
    closing = read_fixtures([season for season in SEASONS if season >= '1920'], sources, extra_columns=CLOSING)
    frame = build_features(fixtures).merge(opponent_form(fixtures), on=KEYS, validate='one_to_one')
    frame = frame.merge(closing[KEYS + CLOSING], on=KEYS, validate='one_to_one')
    if not frame[CLOSING].gt(1).all().all():
        raise ValueError('Invalid closing odds')
    frame[ODDS_COLUMNS[0]] = np.log(frame.AvgCD / frame.AvgCA)
    frame[ODDS_COLUMNS[1]] = np.log(frame.AvgCD / frame.AvgCH)
    development = frame.loc[frame.Season.lt(HOLDOUTS[0])].reset_index(drop=True)
    test = frame.loc[frame.Season.isin(HOLDOUTS)].reset_index(drop=True)
    if len(test) != REFERENCE_MATCHES:
        raise ValueError('Closing odds changed the evaluation cohort')
    folds = [(development.loc[development.Season.lt(season)], development.loc[development.Season.eq(season)]) for season in VALIDATION]
    if any(train.empty or val.empty or train.MatchDate.max() >= val.MatchDate.min() for train, val in folds):
        raise ValueError('Invalid chronological market validation')
    validation = pd.concat([val for _, val in folds]).reset_index(drop=True)
    trials, forecasts = [], {}
    for number, config in enumerate(configurations()):
        values = []
        for train, val in folds:
            fitted = fit_model(train, config)
            values.append(probabilities(fitted, val[FEATURES[config['features']]]))
        forecast = np.vstack(values)
        score = validation_score(validation, forecast)
        name = f'candidate_{number:02d}'
        trials.append({'name': name, 'config': config, **score})
        forecasts[name] = forecast
        print(name, config['family'], config['features'], 'precision', round(score['precision'], 4), flush=True)
    selected = max(trials, key=lambda row: (row['precision'], row['minimum_season_precision'], -row['log_loss']))
    earlier = forecasts[selected['name']]
    cutoff = float(np.quantile(earlier.max(axis=1), 1 - COVERAGE, method='higher'))
    market_validation = market_probability(validation)
    market_score = validation_score(validation, market_validation)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {'target_precision': 0.70, 'target_calls': TARGET_CALLS, 'test_matches': len(test),
                'training_matches': len(development), 'validation_seasons': VALIDATION,
                'test_seasons': HOLDOUTS, 'selected_model': selected, 'fixed_cutoff': cutoff,
                'closing_market_validation': market_score,
                'forecast_time': 'After the archived consensus closing odds are available and before kickoff.',
                'odds_columns': CLOSING,
                'selection': 'Greatest validation precision at 150/619 coverage per season, then worst seasonal precision, then lowest log loss.',
                'evaluation_status': 'Retrospective follow-up on previously examined seasons. Independent future evaluation is required.'}
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    pd.DataFrame([{**row, 'config': json.dumps(row['config'], sort_keys=True)} for row in trials]).to_csv(OUTPUT / 'validation_search.csv', index=False)
    print('Frozen market-aware settings', selected, flush=True)
    fitted = fit_model(development, selected['config'])
    model_dir = ROOT / 'models/precision150'
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / 'market_forecaster.joblib'
    joblib.dump({'model': fitted, 'config': selected['config'], 'columns': FEATURES[selected['config']['features']],
                 'fixed_cutoff': cutoff, 'initial_confidence_history': earlier.max(axis=1)[-350:].tolist()}, path)
    probs = probabilities(fitted, test[FEATURES[selected['config']['features']]])
    restored = joblib.load(path)
    np.testing.assert_allclose(probabilities(restored['model'], test[restored['columns']]), probs, rtol=1e-12, atol=1e-12)
    test_forecasts = {'trained_market_model': probs, 'closing_market_reference': market_probability(test)}
    exports = test[KEYS + ['FTR']].copy()
    metrics = []
    for name, forecast in test_forecasts.items():
        confidence = forecast.max(axis=1)
        selections = {'all_fixtures': np.ones(len(test), dtype=bool), 'top_150_batch': rank_selection(confidence, TARGET_CALLS)}
        if name == 'trained_market_model':
            rolling, thresholds = rolling_selection(test.MatchDate, confidence, earlier.max(axis=1)[-350:], coverage=COVERAGE)
            selections['fixed_cutoff'] = confidence >= cutoff
            selections['rolling_cutoff'] = rolling
            exports['rolling_cutoff_value'] = thresholds
        for index, label in enumerate(CLASSES):
            exports[name + '_p_' + label] = forecast[:, index]
        for policy, picked in selections.items():
            exports[name + '_' + policy] = picked
        metrics.extend(evaluate(test, name, forecast, selections))
    shared_calls = rank_selection(probs.max(axis=1), TARGET_CALLS)
    metrics.extend(evaluate(test, 'closing_market_reference', test_forecasts['closing_market_reference'], {'same_trained_model_fixtures': shared_calls}))
    exports.to_csv(OUTPUT / 'predictions.csv', index=False)
    result = pd.DataFrame(metrics)
    result.to_csv(OUTPUT / 'metrics.csv', index=False)
    print(result.loc[result.season.eq('pooled'), ['model', 'policy', 'covered', 'correct', 'accuracy', 'coverage', 'log_loss']].to_string(index=False))


if __name__ == '__main__':
    main()
