import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from common import CLASSES, ROOT, write_json
from goalkeeper_data import KEYS, KEEPER_COLUMNS, goalkeeper_features, load_keepers
from model import pipeline, probabilities, tune, weekly_bootstrap
from precision_study import FEATURE_SETS, HISTORY_SIGNALS, SEASON_CODES, match_rows, precision_row, read_fixtures

SEASONS = [*SEASON_CODES, '1920']
HOLDOUT = SEASONS[-1]
STEPS = 8
EPOCHS = [10, 25, 40]
SEED = 244
STATUS_LIMIT = 0.05


class FormResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(len(HISTORY_SIGNALS), hidden_size=8, batch_first=True)
        self.form = nn.Linear(8, 1)
        self.correction = nn.Linear(3, len(CLASSES), bias=False)
        nn.init.zeros_(self.correction.weight)

    def forward(self, sequence, length, base_logits):
        batch = sequence.shape[0]
        packed = pack_padded_sequence(sequence.flatten(0, 1), length.flatten(), batch_first=True, enforce_sorted=False)
        _, state = self.gru(packed)
        form = self.form(state[0]).tanh().reshape(batch, 2)
        features = torch.cat([form, (form[:, 0] - form[:, 1]).abs().unsqueeze(1)], dim=1)
        return base_logits + self.correction(features), form


def sequence_scaler(frame):
    rows = np.concatenate([np.asarray(value, dtype=np.float32)
                           for side in ['home', 'away'] for value in frame[side + '_sequence']])
    return rows.mean(axis=0), np.maximum(rows.std(axis=0), 1e-6)


def pack_sequences(frame, mean, std):
    sequence = np.zeros((len(frame), 2, STEPS, len(HISTORY_SIGNALS)), dtype=np.float32)
    length = np.zeros((len(frame), 2), dtype=np.int64)
    for side, team in enumerate(['home', 'away']):
        for i, values in enumerate(frame[team + '_sequence']):
            length[i, side] = len(values)
            sequence[i, side, :len(values)] = (np.asarray(values, dtype=np.float32) - mean) / std
    return torch.from_numpy(sequence), torch.from_numpy(length)


def future_form_targets(frame):
    actual_home = frame.FTR.map({'H': 1.0, 'D': 0.5, 'A': 0.0})
    residual = (actual_home - frame.elo_expected_home).to_numpy(dtype='float32')
    return torch.from_numpy(np.stack([residual, -residual], axis=1))


def fit_form(train, evaluation, columns, params, epochs, seed, checkpoint_path=None):
    torch.manual_seed(seed)
    static = pipeline(columns, 'logistic', params).fit(train, train.FTR)
    train_logits = torch.from_numpy(np.log(probabilities(static, train).clip(min=1e-9)).astype('float32'))
    eval_logits = torch.from_numpy(np.log(probabilities(static, evaluation).clip(min=1e-9)).astype('float32'))
    mean, std = sequence_scaler(train)
    train_x, train_n = pack_sequences(train, mean, std)
    eval_x, eval_n = pack_sequences(evaluation, mean, std)
    targets = torch.tensor([CLASSES.index(label) for label in train.FTR])
    residual = future_form_targets(train)
    model = FormResidual()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.03)
    checkpoints = {}
    for epoch in range(1, max(epochs) + 1):
        model.train()
        for batch in torch.randperm(len(train)).split(128):
            optimizer.zero_grad()
            logits, form = model(train_x[batch], train_n[batch], train_logits[batch])
            loss = F.cross_entropy(logits, targets[batch]) + 0.75 * F.mse_loss(form, residual[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch in epochs:
            model.eval()
            with torch.no_grad():
                logits, form = model(eval_x, eval_n, eval_logits)
                checkpoints[epoch] = (logits.double().softmax(dim=1).numpy(), form.numpy())
    if checkpoint_path is not None:
        torch.save({'weights': model.state_dict(), 'scale_mean': torch.from_numpy(mean),
                    'scale_std': torch.from_numpy(std), 'steps': STEPS, 'features': HISTORY_SIGNALS,
                    'status_threshold': STATUS_LIMIT, 'classes': CLASSES}, checkpoint_path)
        joblib.dump(static, checkpoint_path.with_suffix('.joblib'))
    return checkpoints


def validate_form(history, folds, columns, params, name):
    details = []
    labels, probabilities_by_epoch = [], {epoch: [] for epoch in EPOCHS}
    for index, (fit, val, season) in enumerate(folds):
        train, evaluation = history.iloc[fit], history.iloc[val]
        labels.extend(evaluation.FTR)
        checkpoints = fit_form(train, evaluation, columns, params, EPOCHS, SEED + index)
        for epoch, (probs, _) in checkpoints.items():
            probabilities_by_epoch[epoch].append(probs)
            details.append({'model': name, 'epoch': epoch,
                            'validation_season': season, 'log_loss': log_loss(evaluation.FTR, probs, labels=CLASSES)})
    losses = {epoch: float(log_loss(labels, np.vstack(parts), labels=CLASSES))
              for epoch, parts in probabilities_by_epoch.items()}
    return min(losses, key=losses.get), losses, details


def form_status(form):
    return np.where(form >= STATUS_LIMIT, 'strong', np.where(form <= -STATUS_LIMIT, 'weak', 'steady'))


def plot_timeline(marks, path, team='Man United'):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    chosen = marks.loc[marks.Team.eq(team)].sort_values('MatchDate')
    dates = pd.to_datetime(chosen.MatchDate)
    fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
    ax.axhspan(-STATUS_LIMIT, STATUS_LIMIT, color='#e9eef3', label='Steady range')
    ax.axhline(0, color='#555a64', linewidth=0.8)
    stretches = dates.diff().dt.days.gt(21).cumsum()
    for _, segment in chosen.groupby(stretches):
        ax.plot(dates.loc[segment.index], segment.FormScore, color='#30577e', linewidth=1.5, zorder=2)
    colors = {'strong': '#167049', 'steady': '#526675', 'weak': '#a53642'}
    for status, color in colors.items():
        mask = chosen.FormStatus.eq(status)
        ax.scatter(dates[mask], chosen.FormScore[mask], color=color, label=status.capitalize(), s=26, zorder=3)
    ax.set(title=f'{team} · pre-match form, 2019–2020', ylabel='Expected form relative to Elo', xlabel='Match date')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(loc='best', ncols=4, fontsize=8)
    fig.savefig(path, dpi=175)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sources = json.loads((ROOT / 'data/goalkeeper_sources.json').read_text())
    fixtures = read_fixtures(SEASONS, sources)
    timeline = match_rows(fixtures, sequence_length=STEPS)
    keeper_data = load_keepers(fixtures, SEASONS)
    enriched = timeline.merge(goalkeeper_features(keeper_data), on=KEYS, validate='one_to_one')
    eligible = enriched.loc[enriched.home_keeper_known & enriched.away_keeper_known].copy()
    coverage = enriched.groupby('Season').size().rename('history_eligible').to_frame()
    coverage['with_both_goalkeepers'] = eligible.groupby('Season').size()
    coverage['opta_fixture_matches_all'] = keeper_data.groupby('Season').match_id.count()

    history = eligible.loc[eligible.Season.lt(HOLDOUT)].reset_index(drop=True)
    test = eligible.loc[eligible.Season.eq(HOLDOUT)].reset_index(drop=True)
    folds = [(np.flatnonzero(history.Season.lt(season)), np.flatnonzero(history.Season.eq(season)), season)
             for season in SEASONS[1:-1]]
    assert all(history.iloc[a].MatchDate.max() < history.iloc[b].MatchDate.min() for a, b, _ in folds)
    assert history.MatchDate.max() < test.MatchDate.min() and len(test) > 300

    comparison_columns = {'static': FEATURE_SETS['all_signals'],
                          'static_plus_keeper': FEATURE_SETS['all_signals'] + KEEPER_COLUMNS}
    output = ROOT / 'reports/keeper_form'
    output.mkdir(parents=True, exist_ok=True)
    models = ROOT / 'models/keeper_form'
    models.mkdir(parents=True, exist_ok=True)
    selection = {}
    all_probabilities, marks = {}, {}
    cv_details = []
    for name, columns in comparison_columns.items():
        params, cv_loss, details = tune(history, columns, 'logistic', folds)
        selection[name] = {'params': params, 'cv_log_loss': cv_loss, 'columns': columns}
        cv_details.extend({'model': name, **row} for row in details)
        fitted = pipeline(columns, 'logistic', params).fit(history, history.FTR)
        all_probabilities[name] = probabilities(fitted, test)
        rnn_name = 'gru_plus_keeper' if name.endswith('_keeper') else 'gru'
        epoch, losses, cv = validate_form(history, folds, columns, params, rnn_name)
        cv_details.extend(cv)
        selection[rnn_name] = {'epochs': epoch, 'cv_log_loss': losses[epoch], 'development_by_epoch': losses,
                               'static_model': name}
        all_probabilities[rnn_name], marks[rnn_name] = fit_form(
            history, test, columns, params, [epoch], SEED, models / (rnn_name + '.pt'))[epoch]
        print(f'{name}: development log loss {cv_loss:.5f}. {rnn_name}: selected {epoch} epochs.', flush=True)

    selected = min(selection, key=lambda key: selection[key]['cv_log_loss'])
    predictions = test[KEYS + ['FTR', 'home_keeper_name', 'away_keeper_name',
                               'home_keeper_recent_club_start_share', 'away_keeper_recent_club_start_share',
                               'home_keeper_changed_since_last_game', 'away_keeper_changed_since_last_game']].copy()
    rows = []
    for name, probs in all_probabilities.items():
        if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1, atol=1e-6):
            raise ValueError(f'Invalid probabilities for {name}')
        rows.append(precision_row(name, 'all_matches', test.FTR.to_numpy(), probs, np.ones(len(test), dtype=bool)))
        for idx, outcome in enumerate(CLASSES):
            predictions[f'{name}_p_{outcome}'] = probs[:, idx]
        if name in marks:
            for position, side in enumerate(['home', 'away']):
                predictions[f'{name}_{side}_form_score'] = marks[name][:, position]
                predictions[f'{name}_{side}_form_status'] = form_status(marks[name][:, position])

    contrasts = {'keeper_vs_static': ('static', 'static_plus_keeper'),
                 'gru_vs_static': ('static', 'gru'),
                 'gru_keeper_vs_static': ('static', 'gru_plus_keeper'),
                 'keeper_with_gru': ('gru', 'gru_plus_keeper')}
    comparisons = {key: weekly_bootstrap(test, all_probabilities[base], all_probabilities[added])
                   for key, (base, added) in contrasts.items()}
    forms = []
    for side, team, opponent in [('home', 'HomeTeam', 'AwayTeam'), ('away', 'AwayTeam', 'HomeTeam')]:
        form = predictions[['MatchDate', team, opponent, f'{side}_keeper_name',
                            f'gru_{side}_form_score', f'gru_{side}_form_status']].copy()
        form.columns = ['MatchDate', 'Team', 'Opponent', 'StartingGoalkeeper', 'FormScore', 'FormStatus']
        form['Venue'] = side
        forms.append(form)
    team_timeline = pd.concat(forms, ignore_index=True).sort_values(['Team', 'MatchDate'])
    team_timeline.to_csv(output / 'team_form_timeline.csv', index=False)
    plot_timeline(team_timeline, output / 'team_form_example.png')
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(cv_details).to_csv(output / 'forward_validation.csv', index=False)
    predictions.to_csv(output / 'predictions_and_team_status.csv', index=False)
    coverage.to_csv(output / 'coverage.csv')
    write_json(output / 'run_manifest.json', {
        'seasons': SEASONS, 'development_matches': len(history), 'holdout_matches': len(test),
        'holdout_season': HOLDOUT, 'selected_by_development_log_loss': selected, 'selection': selection,
        'paired_holdout_comparisons': comparisons, 'sequence_length': STEPS, 'sequence_features': HISTORY_SIGNALS,
        'form_status': {'strong_at_or_above': STATUS_LIMIT, 'weak_at_or_below': -STATUS_LIMIT,
                        'meaning': 'Temporary expected change in result score relative to pre-match Elo expectation.'},
        'form_training_target': 'Next-game score (win 1, draw .5, loss 0) minus the pre-match Elo expectation, by team.',
        'fit': {'seed': SEED, 'device': 'cpu', 'torch': torch.__version__, 'hidden_units': 8,
                'maximum_epochs': EPOCHS, 'auxiliary_weight': 0.75},
        'forecast_time': 'After starting lineups were public. Current keeper identity is the only target-match lineup fact used.'})
    print(f'Fixed common cohort: {len(history)} development, {len(test)} holdout. Development selected {selected}.')
    print(pd.DataFrame(rows)[['model', 'correct', 'accuracy', 'log_loss', 'called_D']].to_string(index=False))


if __name__ == '__main__':
    main()
