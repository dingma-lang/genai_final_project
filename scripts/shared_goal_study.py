import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, ROOT, write_json
from confidence_study import choose_policy, measure
from goal_score_study import COLUMNS, conservative_policy, score_probabilities
from model import weekly_bootstrap
from train_precision65 import HOLDOUTS
from xg_study import KEYS, VALIDATION, build_features, load_matches

OUTPUT = ROOT / 'reports/shared_goals'
MODEL_DIR = ROOT / 'models/shared_goals'
SIDE_SIGNALS = [column.removeprefix('home_') for column in COLUMNS if column.startswith('home_')]


def team_features(frame):
    teams = []
    for side, opponent, at_home, sign in [('home', 'away', 1.0, 1.0), ('away', 'home', 0.0, -1.0)]:
        rows = {'at_home': np.full(len(frame), at_home), 'elo_gap': sign * frame.elo_gap.to_numpy(),
                'absolute_elo_gap': frame.absolute_elo_gap.to_numpy()}
        for prefix, source in [('own', side), ('opponent', opponent)]:
            rows[prefix + '_last3_points'] = frame[source.title() + 'Last3Pts'].to_numpy()
            rows[prefix + '_last3_goal_difference'] = frame[source.title() + 'Last3GD'].to_numpy()
            for signal in SIDE_SIGNALS:
                rows[prefix + '_' + signal] = frame[source + '_' + signal].to_numpy()
        teams.append(pd.DataFrame(rows))
    return pd.concat(teams, ignore_index=True)


def targets(frame, target_kind):
    goals = np.concatenate([frame.FTHG.to_numpy(), frame.FTAG.to_numpy()])
    if target_kind == 'goals':
        return goals
    if target_kind == 'goals_and_xg':
        xg = np.concatenate([frame.home_xg.to_numpy(), frame.away_xg.to_numpy()])
        return 0.5 * (goals + xg)
    raise ValueError('Unknown training target')


def fit_model(frame, family, target_kind, config):
    if family == 'linear':
        fitted = make_pipeline(StandardScaler(), PoissonRegressor(alpha=config['alpha'], max_iter=1500))
    elif family == 'catboost':
        fitted = CatBoostRegressor(**config, loss_function='Poisson', learning_rate=0.03,
                                   l2_leaf_reg=5, random_seed=244, thread_count=2,
                                   allow_writing_files=False, verbose=False)
    else:
        raise ValueError('Unknown model family')
    return fitted.fit(team_features(frame), targets(frame, target_kind))


def predict_rates(fitted, family, frame):
    rows = team_features(frame)
    if family == 'catboost':
        values = fitted.predict(rows, prediction_type='Exponent', thread_count=2)
    else:
        values = fitted.predict(rows)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Invalid predicted goal rates')
    return np.column_stack([values[:len(frame)], values[len(frame):]])


def candidates():
    for family, configs in [
            ('linear', [{'alpha': value} for value in [0.1, 1.0]]),
            ('catboost', [{'depth': depth, 'iterations': count} for depth in [3, 4] for count in [300, 600]])]:
        for target_kind in ['goals', 'goals_and_xg']:
            yield family + '_' + target_kind, family, target_kind, configs


def matched_comparisons(test, forecasts, policies, winner):
    rows = []
    outcomes = test.FTR.to_numpy()
    for selector in [winner, 'catboost_goals']:
        policy = policies[selector]
        if policy is None:
            continue
        selected = forecasts[selector].max(axis=1) >= policy['cutoff']
        count = int(selected.sum())
        for name, probs in forecasts.items():
            if name.endswith('_broader'):
                continue
            correct = np.array(CLASSES)[probs.argmax(axis=1)] == outcomes
            order = np.argsort(-probs.max(axis=1), kind='stable')[:count]
            rows.append({'selector': selector, 'model': name, 'calls': count,
                         'same_fixtures_correct': int(correct[selected].sum()),
                         'same_count_correct': int(correct[order].sum()),
                         'same_count_precision': float(correct[order].mean()) if count else None})
    return pd.DataFrame(rows)


def main():
    fixtures, _, _ = load_matches()
    labels = fixtures[KEYS + ['FTHG', 'FTAG', 'home_xg', 'away_xg']]
    frame = build_features(fixtures).merge(labels, on=KEYS, validate='one_to_one')
    development = frame.loc[frame.Season.lt(HOLDOUTS[0])].reset_index(drop=True)
    # Use identical training fixtures for the actual-goal and mixed-target comparisons.
    training = development.dropna(subset=['home_xg', 'away_xg']).reset_index(drop=True)
    test = frame.loc[frame.Season.isin(HOLDOUTS)].reset_index(drop=True)
    folds = [(training.loc[training.Season.lt(season)], development.loc[development.Season.eq(season)], season)
             for season in VALIDATION]
    if any(train.empty or val.empty or train.MatchDate.max() >= val.MatchDate.min() for train, val, _ in folds):
        raise ValueError('Invalid chronological training split')
    outcomes = np.concatenate([val.FTR.to_numpy() for _, val, _ in folds])
    seasons = np.concatenate([val.Season.to_numpy() for _, val, _ in folds])
    selected, validation_rows, cutoffs = {}, [], []
    for name, family, target_kind, configs in candidates():
        trials = []
        for config in configs:
            predictions = []
            for train, val, season in folds:
                fitted = fit_model(train, family, target_kind, config)
                forecast = score_probabilities(predict_rates(fitted, family, val), 0)
                predictions.append(forecast)
                validation_rows.append({'model': name, 'config': json.dumps(config), 'season': season,
                                        'training_matches': len(train), 'training_team_rows': 2 * len(train),
                                        'validation_matches': len(val),
                                        'log_loss': float(log_loss(val.FTR, forecast, labels=CLASSES))})
            forecast = np.vstack(predictions)
            trials.append((float(log_loss(outcomes, forecast, labels=CLASSES)), config, forecast))
        loss, config, forecast = min(trials, key=lambda trial: trial[0])
        conservative, conservative_grid = conservative_policy(outcomes, seasons, forecast)
        broader, broader_grid = choose_policy(outcomes, seasons, forecast, forecast.max(axis=1))
        selected[name] = {'family': family, 'target': target_kind, 'config': config, 'validation_log_loss': loss,
                          'conservative_policy': conservative, 'broader_policy': broader}
        cutoffs.extend({'model': name, 'policy': 'conservative', **row} for row in conservative_grid)
        cutoffs.extend({'model': name, 'policy': 'broader', **row} for row in broader_grid)
        print(name, config, 'validation loss', round(loss, 4), 'conservative cutoff', conservative['cutoff'] if conservative else None, flush=True)
    winner = min(selected, key=lambda name: selected[name]['validation_log_loss'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'selected_by_validation_log_loss': winner, 'models': selected,
                'development_matches': len(development), 'training_matches': len(training),
                'training_team_rows': 2 * len(training), 'test_matches': len(test),
                'validation_seasons': VALIDATION, 'test_seasons': HOLDOUTS,
                'features': team_features(test.iloc[:1]).columns.tolist(),
                'training_target': 'Actual goals or the fixed equal-weight mean of actual goals and archived expected goals.',
                'timing': 'Predictors contain only prior-match summaries and venue. Current-match goals and xG enter training targets only.',
                'shared_model': 'The same scoring function applies to both teams, using own and opponent features plus a home indicator.',
                'evaluation_status': 'Retrospective follow-up on previously examined seasons. Independent prospective evaluation is still required.'}
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    pd.DataFrame(validation_rows).to_csv(OUTPUT / 'forward_validation.csv', index=False)
    pd.DataFrame(cutoffs).to_csv(OUTPUT / 'validation_cutoffs.csv', index=False)
    exported = test[KEYS + ['FTR']].copy()
    forecasts, policies = {}, {}
    for name, settings in selected.items():
        fitted = fit_model(training, settings['family'], settings['target'], settings['config'])
        bundle = {'model': fitted, **settings}
        path = MODEL_DIR / (name + '.joblib')
        joblib.dump(bundle, path)
        rates = predict_rates(fitted, settings['family'], test)
        restored = joblib.load(path)
        np.testing.assert_allclose(predict_rates(restored['model'], restored['family'], test), rates, rtol=1e-12, atol=1e-12)
        forecast = score_probabilities(rates, 0)
        forecasts[name] = forecast
        policies[name] = settings['conservative_policy']
        for index, label in enumerate(CLASSES):
            exported[name + '_p_' + label] = forecast[:, index]
        exported[name + '_covered'] = forecast.max(axis=1) >= policies[name]['cutoff'] if policies[name] else False
        broader_name = name + '_broader'
        forecasts[broader_name] = forecast
        policies[broader_name] = settings['broader_policy']
        exported[broader_name + '_covered'] = forecast.max(axis=1) >= policies[broader_name]['cutoff'] if policies[broader_name] else False
    previous = pd.read_csv(ROOT / 'reports/goal_score/predictions.csv', parse_dates=['MatchDate'], dtype={'Season': str})
    reference = test[KEYS].merge(previous, on=KEYS, how='left', validate='one_to_one')
    old_policy = json.loads((ROOT / 'reports/goal_score/frozen_policy.json').read_text())['policies']['goal_score']
    forecasts['previous_separate_goal_model'] = reference[['goal_score_p_' + label for label in CLASSES]].to_numpy()
    policies['previous_separate_goal_model'] = old_policy
    for name in ['previous_catboost', 'previous_market_benchmark']:
        forecasts[name] = reference[[name + '_p_' + label for label in CLASSES]].to_numpy()
    if not np.isfinite(forecasts['previous_separate_goal_model']).all():
        raise ValueError('Missing paired reference predictions')
    scores = {name: value.max(axis=1) for name, value in forecasts.items()}
    result = measure(test, forecasts, scores, policies)
    result.to_csv(OUTPUT / 'metrics.csv', index=False)
    matched_comparisons(test, forecasts, policies, winner).to_csv(OUTPUT / 'matched_comparisons.csv', index=False)
    exported.to_csv(OUTPUT / 'predictions.csv', index=False)
    comparison = weekly_bootstrap(test, forecasts['previous_separate_goal_model'], forecasts[winner])
    comparison['limitation'] = 'Exploratory weekly interval across previously examined seasons; model search and all serial dependence are not captured.'
    manifest['selected_minus_previous_goal_model'] = comparison
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    print('Selected model', winner, flush=True)
    print(result.loc[result.season.eq('pooled'), ['model', 'policy', 'covered', 'correct', 'accuracy', 'coverage', 'log_loss']].to_string(index=False))


if __name__ == '__main__':
    main()
