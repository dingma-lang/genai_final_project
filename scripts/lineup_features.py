from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from goalkeeper_data import KEYS, load_opta_matches

KEEPER_SIGNALS = ['keeper_changed', 'keeper_recent_share']
OUTFIELD_SIGNALS = ['outfield_changes', 'regular_outfield_missing',
                    'center_back_change_share', 'center_back_comparable']
KEEPER_COLUMNS = [f'{side}_{col}' for side in ['home', 'away'] for col in KEEPER_SIGNALS]
OUTFIELD_COLUMNS = [f'{side}_{col}' for side in ['home', 'away'] for col in OUTFIELD_SIGNALS]


@dataclass(frozen=True)
class StartingXI:
    keeper: str
    outfield: frozenset
    center_backs: frozenset


def load_lineups(fixtures, seasons):
    matched, path, filters = load_opta_matches(fixtures, seasons)
    starters = pd.read_parquet(path, filters=filters + [('is_starter', '=', True)],
                               columns=['match_id', 'team_position', 'player_id', 'position', 'position_side'])
    lookup = {}
    for key, side in starters.groupby(['match_id', 'team_position'], sort=False):
        is_keeper = side.position.eq('Goalkeeper')
        if len(side) != 11 or side.player_id.isna().any() or side.player_id.nunique() != 11 or is_keeper.sum() != 1:
            continue
        is_center_back = side.position.eq('Defender') & side.position_side.str.contains('Centre', case=False, na=False)
        lookup[key] = StartingXI(
            side.loc[is_keeper, 'player_id'].iloc[0],
            frozenset(side.loc[~is_keeper, 'player_id']),
            frozenset(side.loc[is_center_back, 'player_id']),
        )
    return matched[KEYS + ['match_id']], lookup


def side_features(current, history, prefix):
    if current is None or not history or history[-1] is None:
        return None
    recent = [xi for xi in history[-6:] if xi is not None]
    if len(recent) < 3:
        return None
    last = history[-1]
    counts = Counter(player for xi in recent for player in xi.outfield)
    regular = {player for player, count in counts.items() if 3 * count >= 2 * len(recent)}
    comparable = bool(current.center_backs and last.center_backs)
    changed = 0.0
    if comparable:
        changed = 1 - len(current.center_backs & last.center_backs) / max(len(current.center_backs), len(last.center_backs))
    return {
        prefix + '_keeper_changed': int(current.keeper != last.keeper),
        prefix + '_keeper_recent_share': sum(xi.keeper == current.keeper for xi in recent) / len(recent),
        prefix + '_outfield_changes': len(last.outfield - current.outfield),
        prefix + '_regular_outfield_missing': len(regular - current.outfield),
        prefix + '_center_back_change_share': changed,
        prefix + '_center_back_comparable': int(comparable),
    }


def build_lineup_features(fixtures, lookup):
    rows = []
    for season, season_games in fixtures.groupby('Season', sort=True):
        club_history = defaultdict(list)
        for date, day in season_games.groupby('MatchDate', sort=True):
            updates = []
            for game in day.itertuples(index=False):
                row = {key: getattr(game, key) for key in KEYS}
                for side, club in [('home', game.HomeTeam), ('away', game.AwayTeam)]:
                    current = lookup.get((game.match_id, side))
                    row[side + '_starting_xi_known'] = current is not None
                    signals = side_features(current, club_history[club], side)
                    if signals:
                        row.update(signals)
                    updates.append((club, current))
                rows.append(row)
            if len(updates) != len({club for club, _ in updates}):
                raise ValueError(f'A club played twice on {date.date()}')
            for club, current in updates:
                club_history[club].append(current)
    return pd.DataFrame(rows)
