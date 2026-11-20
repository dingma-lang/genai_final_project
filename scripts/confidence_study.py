import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, ROOT, write_json
from goal_score_study import COLUMNS, fit_scores, predict_scores, score_probabilities
from model import probabilities, weekly_bootstrap
from precision_study import precision_row
from train_precision65 import HOLDOUTS
from xg_study import BASE_COLUMNS, KEYS, build_features, fit_candidate, load_matches

OUTPUT = ROOT / 'reports/confidence'
MODEL_DIR = ROOT / 'models/confidence'
OOF_SEASONS = ['1718', '1819', '1920', '2021', '2122', '2223', '2324']
POLICY_SEASONS = OOF_SEASONS[3:]
NAMES = ['goal_history', 'goal_xg', 'catboost_xg', 'blend', 'calibrated_blend', 'blend_confidence_model']


def calibration_inputs(probs):
    clipped = np.clip(probs, 1e-8, 1)
    return np.log(clipped[:, [0, 2]] / clipped[:, [1]])


def confidence_inputs(goal, cat):
    blend = (goal + cat) / 2
    ordered = np.sort(blend, axis=1)
    winners = blend.argmax(axis=1)
    row = np.arange(len(blend))
    return np.column_stack([ordered[:, -1], ordered[:, -1] - ordered[:, -2], blend[:, 1],
                            np.abs(goal[row, winners] - cat[row, winners]), winners == 2,
                            goal.argmax(axis=1) == cat.argmax(axis=1)])


def fit_adjustments(outcomes, goal, cat):
    blend = (goal + cat) / 2
    calibrator = LogisticRegression(C=0.1, max_iter=2000, random_state=244)
    calibrator.fit(calibration_inputs(blend), outcomes)
    correct = (np.array(CLASSES)[blend.argmax(axis=1)] == outcomes).astype(int)
    confidence = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=244))
    confidence.fit(confidence_inputs(goal, cat), correct)
    return calibrator, confidence


def forecast_variants(base, calibrator, confidence):
    result = dict(base)
    result['blend'] = (base['goal_xg'] + base['catboost_xg']) / 2
    result['calibrated_blend'] = probabilities(calibrator, calibration_inputs(result['blend']))
    result['blend_confidence_model'] = result['blend']
    scores = {name: value.max(axis=1) for name, value in result.items()}
    scores['blend_confidence_model'] = confidence.predict_proba(confidence_inputs(base['goal_xg'], base['catboost_xg']))[:, 1]
    return result, scores


def wilson_interval(correct, count):
    if count == 0:
        return None, None
    z = 1.959963984540054
    observed = correct / count
    denominator = 1 + z * z / count
    center = (observed + z * z / (2 * count)) / denominator
    margin = z * np.sqrt(observed * (1 - observed) / count + z * z / (4 * count * count)) / denominator
    return float(center - margin), float(center + margin)


def choose_policy(outcomes, seasons, probs, scores):
    correct = np.array(CLASSES)[probs.argmax(axis=1)] == outcomes
    candidates = []
    for cutoff in np.linspace(0.45, 0.90, 91):
        selected = scores >= cutoff
        count = int(selected.sum())
        if count == 0:
            continue
        hits = int(correct[selected].sum())
        lower, upper = wilson_interval(hits, count)
        row = {'cutoff': float(cutoff), 'calls': count, 'correct': hits, 'precision': hits / count,
               'coverage': float(selected.mean()), 'wilson_lower': lower, 'wilson_upper': upper}
        eligible = lower >= 0.65 and count >= 100 and row['coverage'] >= 0.15
        for season in POLICY_SEASONS[-2:]:
            cohort = seasons == season
            called = selected & cohort
            coverage = float(called.sum() / cohort.sum())
            rate = float(correct[called].mean()) if called.any() else 0.0
            row[season + '_precision'] = rate
            row[season + '_coverage'] = coverage
            eligible &= rate >= 0.65 and coverage >= 0.10
        row['eligible'] = bool(eligible)
        candidates.append(row)
    feasible = [row for row in candidates if row['eligible']]
    policy = max(feasible, key=lambda row: (row['coverage'], row['precision'])) if feasible else None
    return policy, candidates


def fit_base(train, cat_config):
    return {'goal_history': fit_scores(train, 0.1, None, BASE_COLUMNS),
            'goal_xg': fit_scores(train, 0.1, None),
            'catboost_xg': fit_candidate(train, cat_config['family'], cat_config['columns'], cat_config['settings'])}


def predict_base(models, frame):
    return {'goal_history': score_probabilities(predict_scores(models['goal_history'], frame, BASE_COLUMNS), 0),
            'goal_xg': score_probabilities(predict_scores(models['goal_xg'], frame), 0),
            'catboost_xg': probabilities(models['catboost_xg'], frame[COLUMNS])}


def predict_bundle(bundle, frame):
    base = predict_base(bundle['base'], frame)
    return forecast_variants(base, bundle['calibrator'], bundle['confidence'])


def measure(test, forecasts, scores, policies):
    rows = []
    for name, forecast in forecasts.items():
        for season in [*HOLDOUTS, 'pooled']:
            cohort = test.Season.eq(season).to_numpy() if season != 'pooled' else np.ones(len(test), dtype=bool)
            selections = {'all_fixtures': np.ones(int(cohort.sum()), dtype=bool)}
            if policies.get(name):
                selections['selected'] = scores[name][cohort] >= policies[name]['cutoff']
            for policy, selected in selections.items():
                outcomes = test.FTR.to_numpy()[cohort]
                probs = forecast[cohort]
                result = precision_row(name, policy, outcomes, probs, selected)
                result.update({'season': season, 'eligible_matches': int(cohort.sum())})
                result['wilson_lower'], result['wilson_upper'] = wilson_interval(result['correct'], result['covered'])
                for label in CLASSES:
                    actual = outcomes == label
                    predicted = np.array(CLASSES)[probs.argmax(axis=1)]
                    result['recall_' + label] = float((selected & actual & (predicted == label)).sum() / actual.sum())
                rows.append(result)
    return pd.DataFrame(rows)


def main():
    fixtures, _, _ = load_matches()
    frame = build_features(fixtures).merge(fixtures[KEYS + ['FTHG', 'FTAG']], on=KEYS, validate='one_to_one')
    development = frame.loc[frame.Season.lt(HOLDOUTS[0])].reset_index(drop=True)
    test = frame.loc[frame.Season.isin(HOLDOUTS)].reset_index(drop=True)
    previous = json.loads((ROOT / 'reports/xg/frozen_policy.json').read_text())
    cat_config = previous['models']['catboost_xg']
    oof_frames = []
    oof_predictions = {name: [] for name in ['goal_history', 'goal_xg', 'catboost_xg']}
    for season in OOF_SEASONS:
        train = development.loc[development.Season.lt(season)]
        validation = development.loc[development.Season.eq(season)]
        if train.MatchDate.max() >= validation.MatchDate.min():
            raise ValueError('Invalid base-model training chronology')
        models = fit_base(train, cat_config)
        forecasts = predict_base(models, validation)
        oof_frames.append(validation)
        for name, values in forecasts.items():
            oof_predictions[name].append(values)
        print('Completed earlier-match forecasts for', season, flush=True)
    oof = pd.concat(oof_frames).reset_index(drop=True)
    base_forecasts = {name: np.vstack(parts) for name, parts in oof_predictions.items()}
    validation_frames, validation_forecasts, validation_scores = [], {}, {}
    for season in POLICY_SEASONS:
        earlier = oof.Season.lt(season).to_numpy()
        current = oof.Season.eq(season).to_numpy()
        if oof.loc[earlier, 'MatchDate'].max() >= oof.loc[current, 'MatchDate'].min():
            raise ValueError('Calibration uses later outcomes')
        calibrator, confidence = fit_adjustments(oof.FTR.to_numpy()[earlier], base_forecasts['goal_xg'][earlier], base_forecasts['catboost_xg'][earlier])
        forecasts, scores = forecast_variants({name: values[current] for name, values in base_forecasts.items()}, calibrator, confidence)
        validation_frames.append(oof.loc[current])
        for name in NAMES:
            validation_forecasts.setdefault(name, []).append(forecasts[name])
            validation_scores.setdefault(name, []).append(scores[name])
    validation = pd.concat(validation_frames).reset_index(drop=True)
    settings, cutoff_rows, validation_rows = {}, [], []
    for name in NAMES:
        forecast = np.vstack(validation_forecasts[name])
        score = np.concatenate(validation_scores[name])
        policy, grid = choose_policy(validation.FTR.to_numpy(), validation.Season.to_numpy(), forecast, score)
        loss = float(log_loss(validation.FTR, forecast, labels=CLASSES))
        settings[name] = {'validation_log_loss': loss, 'policy': policy}
        cutoff_rows.extend({'model': name, **row} for row in grid)
        validation_rows.append({'model': name, 'log_loss': loss, 'accuracy': float((np.array(CLASSES)[forecast.argmax(axis=1)] == validation.FTR).mean()),
                                'coverage': policy['coverage'] if policy else 0.0, 'precision': policy['precision'] if policy else None})
    feasible = [name for name in NAMES if settings[name]['policy']]
    winner = max(feasible, key=lambda name: (settings[name]['policy']['coverage'], -settings[name]['validation_log_loss'])) if feasible else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'base_forecast_seasons': OOF_SEASONS, 'policy_validation_seasons': POLICY_SEASONS,
                'test_seasons': HOLDOUTS, 'test_matches': len(test), 'selected_by_validation_coverage': winner,
                'settings': settings, 'catboost_settings': cat_config['settings'],
                'poisson_settings': {'alpha': 0.1, 'half_life_days': None, 'rho': 0},
                'calibration': 'Logistic calibration C=0.1 on blended log probability ratios; fitted to earlier-season out-of-fold predictions.',
                'confidence_model': 'Standardized logistic regression C=0.1 on maximum probability, margin, draw probability, model disagreement, home call indicator and model agreement.',
                'policy_requirements': 'Pooled two-sided 95% Wilson lower bound at least 65%, at least 100 calls and 15% coverage; 65% precision and 10% coverage in each of the last two validation seasons.',
                'selection_limitation': 'Wilson bounds are descriptive after searching models and thresholds and assume independent matches.',
                'evaluation_status': 'Retrospective extension on previously examined seasons. Base settings came from earlier experiments. Independent future evaluation still required.'}
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    pd.DataFrame(cutoff_rows).to_csv(OUTPUT / 'validation_cutoffs.csv', index=False)
    pd.DataFrame(validation_rows).to_csv(OUTPUT / 'forward_validation.csv', index=False)
    print('Frozen selection', winner, flush=True)
    calibrator, confidence = fit_adjustments(oof.FTR.to_numpy(), base_forecasts['goal_xg'], base_forecasts['catboost_xg'])
    bundle = {'base': fit_base(development, cat_config), 'calibrator': calibrator, 'confidence': confidence,
              'selected_model': winner, 'policies': {name: settings[name]['policy'] for name in NAMES},
              'features': COLUMNS, 'classes': CLASSES}
    joblib.dump(bundle, MODEL_DIR / 'forecasters.joblib')
    forecasts, scores = predict_bundle(bundle, test)
    reloaded, reloaded_scores = predict_bundle(joblib.load(MODEL_DIR / 'forecasters.joblib'), test)
    for name in NAMES:
        np.testing.assert_allclose(reloaded[name], forecasts[name], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(reloaded_scores[name], scores[name], rtol=1e-12, atol=1e-12)
    result = measure(test, forecasts, scores, bundle['policies'])
    result.to_csv(OUTPUT / 'metrics.csv', index=False)
    exported = test[KEYS + ['FTR']].copy()
    for name, values in forecasts.items():
        for index, label in enumerate(CLASSES):
            exported[name + '_p_' + label] = values[:, index]
        exported[name + '_confidence'] = scores[name]
        policy = bundle['policies'][name]
        exported[name + '_covered'] = scores[name] >= policy['cutoff'] if policy else False
    exported.to_csv(OUTPUT / 'predictions.csv', index=False)
    comparison = weekly_bootstrap(test, forecasts['goal_history'], forecasts['goal_xg'])
    comparison['limitation'] = 'Exploratory weekly interval over two previously examined seasons; model search and all serial dependence are not captured.'
    manifest['xg_minus_history_goal_model'] = comparison
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    print(pd.DataFrame(validation_rows).to_string(index=False), flush=True)
    print(result.loc[result.season.eq('pooled'), ['model', 'policy', 'covered', 'correct', 'accuracy', 'coverage', 'log_loss', 'wilson_lower']].to_string(index=False))


if __name__ == '__main__':
    main()
