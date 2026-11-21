import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import ALLOWED, BASE, CLASSES, ROOT, TRAIN_SEASONS, TEST_SEASON, digest, load_analysis, load_feature_values, load_records, read_inputs, write_json

SENTIMENT = [f'{side}_sentiment_{suffix}' for side in ['home', 'away'] for suffix in ['score', 'unknown']]
CONTEXT = [f'{side}_{key}' for side in ['home', 'away'] for key in ALLOWED if key != 'sentiment']


def feature_sets(df, llm):
    groups = {'baseline': BASE}
    if llm:
        for side in ['home', 'away']:
            source = df[f'{side}_sentiment']
            df[f'{side}_sentiment_unknown'] = source.eq('unknown').astype(int)
            df[f'{side}_sentiment_score'] = pd.to_numeric(source, errors='coerce').fillna(0)
        groups.update({'llm_only': SENTIMENT + CONTEXT, 'combined': BASE + SENTIMENT + CONTEXT,
                       'baseline_plus_sentiment': BASE + SENTIMENT, 'baseline_plus_context': BASE + CONTEXT})
    return groups


def pipeline(columns, family, params):
    numeric = [c for c in columns if c not in CONTEXT]
    cats = [c for c in columns if c in CONTEXT]
    transforms = []
    if numeric:
        transforms.append(('numeric', StandardScaler(), numeric))
    if cats:
        categories = [ALLOWED[c.split('_', 1)[1]] for c in cats]
        transforms.append(('categorical', OneHotEncoder(categories=categories, handle_unknown='error', sparse_output=False), cats))
    prep = ColumnTransformer(transforms, remainder='drop', verbose_feature_names_out=False)
    if family == 'logistic':
        estimator = LogisticRegression(C=params['C'], solver='lbfgs', max_iter=3000, random_state=244)
    else:
        estimator = RandomForestClassifier(n_estimators=300, max_depth=params['max_depth'],
                                            min_samples_leaf=params['min_samples_leaf'], random_state=244, n_jobs=1)
    return Pipeline([('preprocess', prep), ('model', estimator)])


def forward_folds(train):
    folds = []
    for val_season in TRAIN_SEASONS[1:]:
        fit = np.flatnonzero(train.Season.lt(val_season).to_numpy())
        val = np.flatnonzero(train.Season.eq(val_season).to_numpy())
        if not len(fit) or not len(val) or train.iloc[fit].MatchDate.max() >= train.iloc[val].MatchDate.min():
            raise ValueError('Invalid forward-season validation fold')
        if set(train.iloc[fit].FTR) != set(CLASSES):
            raise ValueError('Training fold does not contain all three outcome classes')
        folds.append((fit, val, val_season))
    return folds


def probabilities(fitted, frame):
    raw = fitted.predict_proba(frame)
    order = [list(fitted.classes_).index(c) for c in CLASSES]
    output = raw[:, order]
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1):
        raise ValueError('Invalid predicted probabilities')
    return output


def tune(train, columns, family, folds):
    candidates = [{'C': c} for c in [0.01, 0.1, 1.0, 10.0]] if family == 'logistic' else [
        {'max_depth': 3, 'min_samples_leaf': 10}, {'max_depth': 5, 'min_samples_leaf': 10},
        {'max_depth': None, 'min_samples_leaf': 10}]
    candidate_scores = []
    detail = []
    for params in candidates:
        ys, ps = [], []
        for fit, val, season in folds:
            fitted = pipeline(columns, family, params).fit(train.iloc[fit], train.iloc[fit].FTR)
            pred = probabilities(fitted, train.iloc[val])
            ys.extend(train.iloc[val].FTR.tolist())
            ps.append(pred)
            detail.append({'family': family, 'params': json.dumps(params, sort_keys=True), 'validation_season': season,
                           'n_fit': len(fit), 'n_validation': len(val),
                           'validation_log_loss': log_loss(train.iloc[val].FTR, pred, labels=CLASSES)})
        candidate_scores.append(log_loss(ys, np.vstack(ps), labels=CLASSES))
    winner = int(np.argmin(candidate_scores))
    return candidates[winner], candidate_scores[winner], detail


def weekly_bootstrap(test, base, combined, repetitions=2000):
    y_indices = np.array([CLASSES.index(c) for c in test.FTR])
    row = np.arange(len(test))
    base_loss = -np.log(np.clip(base[row, y_indices], np.finfo(float).eps, 1))
    comb_loss = -np.log(np.clip(combined[row, y_indices], np.finfo(float).eps, 1))
    acc_delta = (combined.argmax(axis=1) == y_indices).astype(float) - (base.argmax(axis=1) == y_indices).astype(float)
    week = test.MatchDate.dt.to_period('W-SUN').astype(str).to_numpy()
    blocks = [np.flatnonzero(week == key) for key in sorted(set(week))]
    rng = np.random.default_rng(244)
    loss_deltas, acc_deltas = [], []
    for _ in range(repetitions):
        ids = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        loss_deltas.append(float((comb_loss - base_loss)[ids].mean()))
        acc_deltas.append(float(acc_delta[ids].mean()))
    return {'contrast': 'combined minus baseline; negative log-loss difference favors combined',
            'log_loss_difference': float((comb_loss - base_loss).mean()),
            'log_loss_difference_95pct_interval': np.quantile(loss_deltas, [0.025, 0.975]).tolist(),
            'accuracy_difference': float(acc_delta.mean()),
            'accuracy_difference_95pct_interval': np.quantile(acc_deltas, [0.025, 0.975]).tolist(),
            'bootstrap_repetitions': repetitions, 'calendar_week_blocks': len(blocks),
            'limitation': 'Exploratory calendar-week block bootstrap on one season; does not capture all team/serial dependence.'}


def plot_results(metrics, output):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = metrics[metrics.family.ne('benchmark')].copy()
    names = [f'{r.feature_set}\n{r.family}' for r in show.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(show) * 1.25), 4.8))
    axes[0].bar(names, show.accuracy, color='#326c9b')
    home = float(metrics.loc[metrics.feature_set.eq('always_home'), 'accuracy'].iloc[0])
    axes[0].axhline(home, color='#b44b42', linestyle='--', label=f'Always home: {home:.1%}')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Held-out accuracy (higher is better)')
    axes[0].legend(fontsize=8)
    axes[1].bar(names, show.log_loss, color='#4e8d70')
    prior = float(metrics.loc[metrics.feature_set.eq('train_class_prior'), 'log_loss'].iloc[0])
    axes[1].axhline(prior, color='#b44b42', linestyle='--', label=f'Train-prior log loss: {prior:.3f}')
    axes[1].set_ylabel('Held-out log loss (lower is better)')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=7, rotation=35)
    fig.suptitle('EPL 2017–2018 holdout: fixed cohort of 279 matches')
    fig.tight_layout()
    fig.savefig(output / 'model_comparison.png', dpi=180)
    plt.close(fig)


def run(features=None, human_validation='pending', tag=None):
    df = load_analysis()
    eligible = df.loc[df[BASE].notna().all(axis=1)].copy()
    provenance = None
    if features:
        values, provenance = load_feature_values(features) if features.suffix == '.csv' else load_records(features, read_inputs())
        missing = set(eligible.MatchID) - set(values.MatchID)
        if missing:
            raise ValueError(f'LLM features missing for {len(missing)} eligible matches; the cohort must remain fixed')
        eligible = eligible.merge(values, on='MatchID', how='left', validate='one_to_one')
    groups = feature_sets(eligible, bool(features))
    train = eligible[eligible.Season.isin(TRAIN_SEASONS)].reset_index(drop=True)
    test = eligible[eligible.Season.eq(TEST_SEASON)].reset_index(drop=True)
    if (len(train), len(test)) != (1014, 279):
        raise ValueError('Cohort does not match the documented 1014/279 split')
    folds = forward_folds(train)
    mode = 'full' if features else 'baseline'
    if tag:
        mode += '_' + tag  # e.g. reports/full_L5, so earlier results are not overwritten
    output = ROOT / 'reports' / mode
    models = ROOT / 'models' / mode
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    predictions = test[['MatchID', 'Season', 'MatchDate', 'HomeStd', 'AwayStd', 'FTR']].copy()
    predictions['always_home_prediction'] = 'H'
    prior = (train.FTR.value_counts().reindex(CLASSES, fill_value=0).to_numpy() + 1) / (len(train) + 3)
    prior_p = np.tile(prior, (len(test), 1))
    scores = [{'family': 'benchmark', 'feature_set': 'always_home', 'n_test': len(test),
               'accuracy': float(test.FTR.eq('H').mean()), 'log_loss': None, 'cv_log_loss': None, 'parameters': '{}'},
              {'family': 'benchmark', 'feature_set': 'train_class_prior', 'n_test': len(test),
               'accuracy': accuracy_score(test.FTR, np.array(CLASSES)[prior_p.argmax(axis=1)]),
               'log_loss': log_loss(test.FTR, prior_p, labels=CLASSES), 'cv_log_loss': None, 'parameters': '{}'}]
    fitted_probs, details, coefficients, importances, confusion = {}, [], [], [], []
    for family in ['logistic', 'random_forest']:
        for group, columns in groups.items():
            if family == 'random_forest' and group.startswith('baseline_plus_'):
                continue
            params, cv_loss, cv_rows = tune(train, columns, family, folds)
            details.extend(dict(r, feature_set=group) for r in cv_rows)
            fitted = pipeline(columns, family, params).fit(train, train.FTR)
            probs = probabilities(fitted, test)
            predicted = np.array(CLASSES)[probs.argmax(axis=1)]
            key = family + '__' + group
            fitted_probs[key] = probs
            predictions[key + '_prediction'] = predicted
            for i, outcome in enumerate(CLASSES):
                predictions[key + '_p_' + outcome] = probs[:, i]
            score = {'family': family, 'feature_set': group, 'n_test': len(test),
                     'accuracy': float(accuracy_score(test.FTR, predicted)),
                     'log_loss': float(log_loss(test.FTR, probs, labels=CLASSES)),
                     'cv_log_loss': float(cv_loss), 'parameters': json.dumps(params, sort_keys=True)}
            scores.append(score)
            joblib.dump(fitted, models / (key + '.joblib'))
            names = fitted.named_steps['preprocess'].get_feature_names_out()
            estimator = fitted.named_steps['model']
            if family == 'logistic':
                for outcome, coefs in zip(estimator.classes_, estimator.coef_):
                    coefficients.extend({'feature_set': group, 'outcome': outcome, 'feature': name, 'coefficient': float(value)}
                                        for name, value in zip(names, coefs))
            else:
                importances.extend({'feature_set': group, 'feature': name, 'impurity_importance': float(value)}
                                   for name, value in zip(names, estimator.feature_importances_))
            matrix = confusion_matrix(test.FTR, predicted, labels=CLASSES)
            confusion.extend({'family': family, 'feature_set': group, 'actual': actual, 'predicted': pred, 'count': int(matrix[i, j])}
                             for i, actual in enumerate(CLASSES) for j, pred in enumerate(CLASSES))
            print(f'{family:14} {group:24} accuracy={score["accuracy"]:.3f} log_loss={score["log_loss"]:.4f} params={params}', flush=True)
    metrics = pd.DataFrame(scores)
    metrics.to_csv(output / 'metrics.csv', index=False)
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    pd.DataFrame(details).to_csv(output / 'forward_validation.csv', index=False)
    pd.DataFrame(coefficients).to_csv(output / 'logistic_coefficients.csv', index=False)
    pd.DataFrame(importances).to_csv(output / 'forest_importances.csv', index=False)
    pd.DataFrame(confusion).to_csv(output / 'confusion_matrices.csv', index=False)
    comparisons = {}
    if features:
        for family in ['logistic', 'random_forest']:
            comparisons[family] = weekly_bootstrap(test, fitted_probs[family + '__baseline'], fitted_probs[family + '__combined'])
        write_json(output / 'paired_comparisons.json', comparisons)
    write_json(output / 'run_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'mode': mode,
        'analysis_sha256': digest((ROOT / 'data/processed/previews_with_results.csv').read_text()),
        'feature_file_sha256': digest(features.read_text()) if features else None,
        'llm_provenance': sorted([list(p) for p in provenance]) if provenance else None,
        'code_sha256': {name: digest((ROOT / 'scripts' / name).read_text()) for name in ['model.py', 'common.py']},
        'versions': {'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
        'train_seasons': TRAIN_SEASONS, 'test_season': TEST_SEASON, 'train_n': len(train), 'test_n': len(test),
        'feature_sets': groups, 'seed': 244, 'class_probability_order': CLASSES,
        'selection': 'Minimum pooled forward-validation log loss; ties follow fixed candidate order. Test metrics never enter tuning.',
        'folds': [{'validation_season': season, 'train_n': len(fit), 'validation_n': len(val)} for fit, val, season in folds],
        'interpretation': 'Logistic coefficients use train-standardized numeric inputs and one-hot categories; associations, not causal effects. Forest importance is descriptive and biased.',
        'llm_models_fitted': bool(features), 'extraction_quality_checks_complete': False,
        'human_reference_validation': human_validation})
    plot_results(metrics, output)
    lines = ['# Modeling results', '', f'Training: {len(train)} matches (2013–2017). Test: {len(test)} matches (2017–2018).', '',
             'Hyperparameters were chosen using three expanding-season validation folds inside the training period. Scaling is fitted within each fold. The final season is used for evaluation only.', '',
             '| Model | Features | Test accuracy | Test log loss |', '| --- | --- | ---: | ---: |']
    for r in scores:
        loss = 'not scored' if r['log_loss'] is None else f'{r["log_loss"]:.4f}'
        lines.append(f'| {r["family"]} | {r["feature_set"]} | {r["accuracy"]:.2%} | {loss} |')
    lines.extend(['', 'The deterministic always-home rule is an accuracy benchmark. Its probabilities assign zero to other results, so its theoretical log loss is infinite and is not used as a probability-quality benchmark. The smoothed training class prior supplies the probability benchmark.', ''])
    baseline_predictions = np.array(CLASSES)[fitted_probs['logistic__baseline'].argmax(axis=1)]
    correct = int((baseline_predictions == test.FTR.to_numpy()).sum())
    predicted_draws = int((baseline_predictions == 'D').sum())
    actual_draws = int(test.FTR.eq('D').sum())
    lines.extend([f'The logistic baseline correctly classified {correct}/{len(test)} cases. It predicted {predicted_draws} draws among {actual_draws} actual draws. Small accuracy differences alone should not be interpreted as reliable improvement.', ''])
    if not features:
        lines.append('The LLM-only, combined, and feature-ablation models have not been run because no extracted LLM features are available. These baseline results do not answer whether LLM features help.')
    else:
        lines.extend(['The main hypothesis compares combined versus baseline logistic log loss. See paired_comparisons.json for week-block bootstrap intervals. Compare baseline_plus_sentiment and baseline_plus_context to assess H3; this is an exploratory ablation, not a causal test.', '',
                      'Review masking limitations, repeat-extraction agreement, the exclusion log, and the metadata-only leakage diagnostic alongside these results. Model fitting alone does not establish extraction quality or absence of contamination.'])
        logistic = {row['feature_set']: row for row in scores if row['family'] == 'logistic'}
        home_accuracy = scores[0]['accuracy']
        h1_difference = logistic['llm_only']['accuracy'] - home_accuracy
        h2 = comparisons['logistic']
        h3_difference = logistic['baseline_plus_sentiment']['log_loss'] - logistic['baseline_plus_context']['log_loss']
        interval = h2['log_loss_difference_95pct_interval']
        lines.extend(['', 'Observed hypothesis comparisons (logistic regression):', '',
                      f'- H1: LLM-only accuracy minus always-home accuracy = {h1_difference:+.2%}.',
                      f'- H2: Combined minus baseline log loss = {h2["log_loss_difference"]:+.4f}; exploratory 95% week-block interval [{interval[0]:+.4f}, {interval[1]:+.4f}]. Negative values favor combined.',
                      f'- H3: Baseline-plus-sentiment minus baseline-plus-context log loss = {h3_difference:+.4f}. Negative values favor the sentiment addition. This ablation does not establish general or causal feature importance.'])
        if human_validation == 'omitted':
            lines.extend(['', 'Independent human reference annotation was omitted by the team, departing from the 50-preview hand-labeled pilot proposed in Milestone 1. The classifier targets are actual match outcomes, so model training does not require human feature labels. Automated schema/quotation checks and repeatability do not establish the semantic accuracy of the extracted features.'])
        else:
            lines.extend(['', 'Independent human pilot agreement remains pending and must not be reported as completed.'])
    lines.extend(['', 'Limitations: one historical league and one held-out season; small sample; previews cover a selected subset of fixtures. The test season has now been inspected for this fixed design and must not be used for further model or prompt tuning.', ''])
    (output / 'results.md').write_text('\n'.join(lines))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, help='Extraction JSONL or exported feature CSV with its manifest; omit for baseline only')
    parser.add_argument('--human-validation', choices=['pending', 'omitted'], default='pending')
    parser.add_argument('--tag', help='Suffix for the report and model folders, e.g. L5 writes reports/full_L5')
    args = parser.parse_args()
    if args.tag and not args.tag.replace('_', '').isalnum():
        parser.error('--tag may contain only letters, numbers and underscores')
    run(args.features, args.human_validation, args.tag)
