import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from common import CLASSES, ROOT, write_json
from goalkeeper_data import KEYS
from lineup_features import KEEPER_COLUMNS, OUTFIELD_COLUMNS, build_lineup_features, load_lineups
from model import pipeline, probabilities, tune, weekly_bootstrap
from precision_study import FEATURE_SETS, SEASON_CODES, match_rows, precision_row, read_fixtures

SEASONS = [*SEASON_CODES, '1920', '2021', '2122']
HOLDOUT = SEASONS[-1]
EXTRA_RAW = ['HTHG', 'HTAG', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
EXTRA_PAIRS = {'first_half_for': ('HTHG', 'HTAG'), 'first_half_against': ('HTAG', 'HTHG'),
               'corners_for': ('HC', 'AC'), 'corners_against': ('AC', 'HC'),
               'fouls': ('HF', 'AF'), 'yellows': ('HY', 'AY'), 'reds': ('HR', 'AR')}
ROLLING_GROUPS = {'first_half': ['first_half_for', 'first_half_against'],
                  'corners': ['corners_for', 'corners_against'], 'discipline': ['fouls', 'yellows', 'reds']}
DRAW_COLUMNS = ['scoring_total', 'scoring_gap', 'close_low_scoring', 'recent_draw_share']
BASE_COLUMNS = FEATURE_SETS['all_signals']
DRAW_INDEX = CLASSES.index('D')
MAX_VALIDATION_ACCURACY_DROP = 0.02
THRESHOLD_GRID = [i / 1000 for i in range(150, 405, 5)] + [1.0]


def build_numerical_rows(fixtures):
    schedule = match_rows(fixtures)
    histories = []
    for index in range(2):
        role = ['HomeTeam', 'AwayTeam'][index]
        frame = fixtures[KEYS + ['FTR']].copy()
        frame['club'] = frame[role]
        frame['venue'] = role
        frame['draw'] = frame.FTR.eq('D').astype(float)
        for key, pair in EXTRA_PAIRS.items():
            frame[key] = fixtures[pair[index]]
        histories.append(frame)
    long = pd.concat(histories, ignore_index=True).sort_values(['Season', 'club', 'MatchDate'])
    if long.duplicated(['Season', 'club', 'MatchDate']).any():
        raise ValueError('Club schedule is ambiguous')
    fields = [*EXTRA_PAIRS, 'draw']
    long[fields] = (long.groupby(['Season', 'club'], sort=False)[fields]
                        .transform(lambda series: series.shift(1).rolling(6, min_periods=3).mean()))
    for role, prefix in [('HomeTeam', 'home'), ('AwayTeam', 'away')]:
        additions = long.loc[long.venue.eq(role), KEYS + fields].rename(columns={c: f'{prefix}_{c}' for c in fields})
        schedule = schedule.merge(additions, on=KEYS, validate='one_to_one')

    home_scoring = (schedule.home_venue_goals_for + schedule.away_venue_goals_against) / 2
    away_scoring = (schedule.away_venue_goals_for + schedule.home_venue_goals_against) / 2
    schedule['scoring_total'] = home_scoring + away_scoring
    schedule['scoring_gap'] = (home_scoring - away_scoring).abs()
    elo_closeness = np.exp(-np.abs(schedule.elo_gap + 60) / 150)
    schedule['close_low_scoring'] = elo_closeness * (3 - schedule.scoring_total)
    schedule['recent_draw_share'] = (schedule.home_draw + schedule.away_draw) / 2
    if schedule[[f'{side}_{key}' for side in ['home', 'away'] for key in fields] + DRAW_COLUMNS].isna().any().any():
        raise ValueError('Missing numerical history for an eligible fixture')
    return schedule


def fold_indices(history):
    folds = []
    for season in SEASONS[1:-1]:
        fit = np.flatnonzero(history.Season.lt(season))
        val = np.flatnonzero(history.Season.eq(season))
        if len(fit) == 0 or len(val) == 0 or history.iloc[fit].MatchDate.max() >= history.iloc[val].MatchDate.min():
            raise ValueError(f'Invalid chronology in validation fold {season}')
        folds.append((fit, val, season))
    return folds


def draw_predictions(probs, threshold):
    labels = np.array(CLASSES)[probs.argmax(axis=1)]
    labels[probs[:, DRAW_INDEX] >= threshold] = 'D'
    return labels


def choose_draw_rule(history, folds, columns, params):
    values = []
    labels = []
    for fit, val, _ in folds:
        model = pipeline(columns, 'logistic', params).fit(history.iloc[fit], history.iloc[fit].FTR)
        values.append(probabilities(model, history.iloc[val]))
        labels.extend(history.iloc[val].FTR)
    probs = np.vstack(values)
    outcomes = np.asarray(labels)
    natural_accuracy = (draw_predictions(probs, 1.0) == outcomes).mean()
    acceptable = []
    for threshold in THRESHOLD_GRID:
        predictions = draw_predictions(probs, threshold)
        accuracy = float((predictions == outcomes).mean())
        if accuracy + MAX_VALIDATION_ACCURACY_DROP + 1e-10 >= natural_accuracy:
            draw_f1 = float(f1_score(outcomes == 'D', predictions == 'D', zero_division=0))
            acceptable.append((draw_f1, accuracy, threshold))
    f1, accuracy, threshold = max(acceptable)
    return {'threshold': threshold, 'validation_draw_f1': f1, 'validation_accuracy': accuracy,
            'validation_argmax_accuracy': float(natural_accuracy)}


def draw_ranking(outcomes, probs):
    is_draw = outcomes == 'D'
    draw_prob = probs[:, DRAW_INDEX]
    count = math.ceil(len(draw_prob) / 10)
    top = np.argsort(-draw_prob, kind='stable')[:count]
    return {'draw_average_precision': average_precision_score(is_draw, draw_prob),
            'top_draw_count': count, 'top_draw_correct': int(is_draw[top].sum()),
            'draw_precision_at_top_10pct': float(is_draw[top].mean())}


def run_track(data, groups, draw_variants=()):
    history = data.loc[data.Season.lt(HOLDOUT)].reset_index(drop=True)
    test = data.loc[data.Season.eq(HOLDOUT)].reset_index(drop=True)
    if history.MatchDate.max() >= test.MatchDate.min():
        raise ValueError('Holdout is not later than development')
    folds = fold_indices(history)
    configs = {}
    cv_rows = []
    for name, columns in groups.items():
        params, cv_log_loss, cv = tune(history, columns, 'logistic', folds)
        configs[name] = {'columns': columns, 'params': params, 'validation_log_loss': cv_log_loss}
        cv_rows.extend({'model': name, **row} for row in cv)
    selected = min(configs, key=lambda name: configs[name]['validation_log_loss'])
    policies = {}
    for name in draw_variants:
        policies[name] = choose_draw_rule(history, folds, configs[name]['columns'], configs[name]['params'])

    predictions = test[KEYS + ['FTR']].copy()
    outcomes = test.FTR.to_numpy()
    evaluated = np.ones(len(test), dtype=bool)
    metrics, forecast = [], {}
    for name, config in configs.items():
        model = pipeline(config['columns'], 'logistic', config['params']).fit(history, history.FTR)
        probs = probabilities(model, test)
        forecast[name] = probs
        for index, label in enumerate(CLASSES):
            predictions[f'{name}_p_{label}'] = probs[:, index]
        argmax = draw_predictions(probs, 1.0)
        predictions[name + '_argmax'] = argmax
        rank_stats = draw_ranking(outcomes, probs)
        metrics.append({**precision_row(name, 'argmax', outcomes, probs, evaluated, argmax), **rank_stats})
        if name in policies:
            labels = draw_predictions(probs, policies[name]['threshold'])
            predictions[name + '_draw_rule'] = labels
            metrics.append({**precision_row(name, 'draw_rule', outcomes, probs, evaluated, labels), **rank_stats})
    reference = next(iter(groups))
    comparisons = {name: weekly_bootstrap(test, forecast[reference], forecast[name])
                   for name in groups if name != reference}
    manifest = {'holdout_season': HOLDOUT, 'development_n': len(history), 'holdout_n': len(test),
                'holdout_draws': int((outcomes == 'D').sum()), 'selected_by_development_log_loss': selected,
                'models': configs, 'draw_rules_selected_in_development': policies,
                'versus_reference': comparisons, 'validation_seasons': [s for _, _, s in folds]}
    return pd.DataFrame(metrics), pd.DataFrame(cv_rows), predictions, manifest


def save_track(folder, results):
    folder.mkdir(parents=True, exist_ok=True)
    metrics, cv, predictions, manifest = results
    metrics.to_csv(folder / 'metrics.csv', index=False)
    cv.to_csv(folder / 'forward_validation.csv', index=False)
    predictions.to_csv(folder / 'test_predictions.csv', index=False)
    write_json(folder / 'run_manifest.json', manifest)
    print(f'{folder.name}: {manifest["development_n"]} training, {manifest["holdout_n"]} holdout. '
          f'Development selected {manifest["selected_by_development_log_loss"]}.', flush=True)
    print(metrics[['model', 'policy', 'accuracy', 'log_loss', 'called_D', 'precision_D', 'recall_D',
                   'draw_precision_at_top_10pct']].to_string(index=False), flush=True)


def main():
    sources = [*json.loads((ROOT / 'data/goalkeeper_sources.json').read_text()),
               *json.loads((ROOT / 'data/draw_lineup_sources.json').read_text())]
    fixtures = read_fixtures(SEASONS, sources, extra_columns=EXTRA_RAW)
    schedule = build_numerical_rows(fixtures)
    numeric_additions = {'draw_signals': DRAW_COLUMNS,
                         **{name: [f'{side}_{key}' for side in ['home', 'away'] for key in fields]
                            for name, fields in ROLLING_GROUPS.items()}}
    groups = {'existing_elo': BASE_COLUMNS, **{name: BASE_COLUMNS + fields for name, fields in numeric_additions.items()},
              'all_new_signals': BASE_COLUMNS + [key for fields in numeric_additions.values() for key in fields]}
    output = ROOT / 'reports/draw_lineup'
    day_ahead = run_track(schedule, groups, draw_variants=['existing_elo', 'draw_signals'])
    save_track(output / 'day_ahead', day_ahead)

    matched, lookup = load_lineups(fixtures, SEASONS)
    continuity = build_lineup_features(matched, lookup)
    combined = schedule.merge(continuity, on=KEYS, validate='one_to_one')
    paired = combined.dropna(subset=KEEPER_COLUMNS + OUTFIELD_COLUMNS).copy()
    coverage = combined.groupby('Season').agg(numeric_eligible=('FTR', 'size'),
                                              home_starting_xi=('home_starting_xi_known', 'sum'),
                                              away_starting_xi=('away_starting_xi_known', 'sum'))
    coverage['continuity_eligible'] = paired.groupby('Season').size()
    coverage.to_csv(output / 'lineup_coverage.csv')
    selected_numeric = day_ahead[3]['selected_by_development_log_loss']
    reference = groups[selected_numeric]
    lineup_groups = {'same_fixtures_numeric': reference,
                     'plus_keeper_continuity': reference + KEEPER_COLUMNS,
                     'plus_outfield_continuity': reference + OUTFIELD_COLUMNS,
                     'plus_full_lineup': reference + KEEPER_COLUMNS + OUTFIELD_COLUMNS}
    lineup = run_track(paired, lineup_groups)
    lineup[3]['numerical_reference_selected_in_day_ahead_validation'] = selected_numeric
    lineup[3]['forecast_time'] = 'After both starting XIs are public and before kickoff.'
    save_track(output / 'post_lineup', lineup)


if __name__ == '__main__':
    main()
