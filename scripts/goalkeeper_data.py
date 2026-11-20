import hashlib
import json
from collections import defaultdict

import pandas as pd

from common import ROOT

TEAM_NAMES = {
    'AFC Bournemouth': 'Bournemouth', 'Brighton & Hove Albion': 'Brighton',
    'Cardiff City': 'Cardiff', 'Huddersfield Town': 'Huddersfield', 'Hull City': 'Hull',
    'Leeds United': 'Leeds', 'Leicester City': 'Leicester', 'Manchester City': 'Man City', 'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle', 'Norwich City': 'Norwich', 'Queens Park Rangers': 'QPR',
    'Stoke City': 'Stoke', 'Swansea City': 'Swansea', 'Tottenham Hotspur': 'Tottenham',
    'West Bromwich Albion': 'West Brom', 'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves',
}
KEYS = ['Season', 'MatchDate', 'HomeTeam', 'AwayTeam']
SIGNALS = ['recent_club_start_share', 'changed_since_last_game', 'known_club_starts',
           'prior_full_matches', 'prior_save_proxy']
KEEPER_COLUMNS = [f'{side}_keeper_{signal}' for side in ['home', 'away'] for signal in SIGNALS]


def verify_archive(path, expected_hash):
    with path.open('rb') as source:
        actual = hashlib.file_digest(source, 'sha256').hexdigest()
    if actual != expected_hash:
        raise ValueError(f'Archive hash changed: {path}')


def load_opta_matches(fixtures, season_codes):
    sources = json.loads((ROOT / 'data/goalkeeper_sources.json').read_text())
    archives = {row['path'].split('/')[-1]: row for row in sources if row['path'].endswith('.parquet')}
    for row in archives.values():
        verify_archive(ROOT / row['path'], row['sha256'])
    seasons = {f'20{code[:2]}-20{code[2:]}': code for code in season_codes}
    filters = [('competition', '=', 'EPL'), ('season', 'in', list(seasons))]
    matches = pd.read_parquet(ROOT / archives['opta_fixtures.parquet']['path'], filters=filters,
                              columns=['match_id', 'match_date', 'season', 'match_status',
                                       'home_team', 'away_team', 'home_score', 'away_score'])
    matches = matches.loc[matches.match_status.eq('Played')].copy()
    matches[['home_score', 'away_score']] = matches[['home_score', 'away_score']].apply(pd.to_numeric, errors='raise')
    matches['Season'] = matches.season.map(seasons)
    matches['MatchDate'] = pd.to_datetime(matches.match_date.str[:10], format='%Y-%m-%d')
    for side in ['home', 'away']:
        names = matches[f'{side}_team'].str.replace(r' (?:AFC|FC)$', '', regex=True)
        matches[side.capitalize() + 'Team'] = names.replace(TEAM_NAMES)

    joined = fixtures.merge(matches[KEYS + ['match_id', 'home_score', 'away_score']], how='left',
                            on=KEYS, validate='one_to_one')
    matched = joined.match_id.notna()
    if not joined.loc[matched, 'FTHG'].eq(joined.loc[matched, 'home_score']).all() or not joined.loc[matched, 'FTAG'].eq(joined.loc[matched, 'away_score']).all():
        raise ValueError('Opta and Football-Data scores disagree for matched fixtures')
    return joined, ROOT / archives['opta_lineups.parquet']['path'], filters


def load_keepers(fixtures, season_codes):
    joined, lineups_path, filters = load_opta_matches(fixtures, season_codes)
    starters = pd.read_parquet(lineups_path,
                               filters=filters + [('position', '=', 'Goalkeeper'), ('is_starter', '=', True)],
                               columns=['match_id', 'team_position', 'player_id', 'player_name', 'minutes_played'])
    if starters.duplicated(['match_id', 'team_position']).any():
        raise ValueError('Opta contains more than one starting goalkeeper for a team fixture')
    for side in ['home', 'away']:
        side_rows = starters.loc[starters.team_position.eq(side), ['match_id', 'player_id', 'player_name', 'minutes_played']]
        side_rows = side_rows.rename(columns={key: f'{side}_{key}' for key in side_rows if key != 'match_id'})
        joined = joined.merge(side_rows, how='left', on='match_id', validate='many_to_one')
    return joined.sort_values(['MatchDate', 'HomeTeam'])


def goalkeeper_features(joined):
    rows = []
    keeper_games = defaultdict(list)
    for season, season_rows in joined.groupby('Season', sort=True):
        club_games = defaultdict(list)
        for _, day in season_rows.groupby('MatchDate', sort=True):
            updates = []
            for match in day.itertuples(index=False):
                row = {name: getattr(match, name) for name in KEYS}
                for side, club, conceded, faced in [('home', match.HomeTeam, match.FTAG, match.AST),
                                                    ('away', match.AwayTeam, match.FTHG, match.HST)]:
                    player = getattr(match, side + '_player_id')
                    minutes = getattr(match, side + '_minutes_played')
                    known = pd.notna(player)
                    prefix = side + '_keeper_'
                    row[prefix + 'name'] = getattr(match, side + '_player_name')
                    row[prefix + 'known'] = known
                    if known:
                        recent = club_games[club][-6:]
                        known_recent = [previous for previous in recent if previous is not None]
                        previous_full = keeper_games[player][-10:]
                        shots = sum(s for s, _ in previous_full)
                        proxy_saves = sum(max(s - g, 0) for s, g in previous_full)
                        row[prefix + 'recent_club_start_share'] = known_recent.count(player) / len(known_recent) if known_recent else 0
                        row[prefix + 'changed_since_last_game'] = int(bool(recent) and recent[-1] is not None and recent[-1] != player)
                        row[prefix + 'known_club_starts'] = len(known_recent)
                        row[prefix + 'prior_full_matches'] = len(previous_full)
                        row[prefix + 'prior_save_proxy'] = (proxy_saves + 8 * 0.68) / (shots + 8)
                    updates.append((club, player if known else None, minutes, faced, conceded))
                rows.append(row)
            for club, player, minutes, shots, goals in updates:
                club_games[club].append(player)
                if player is not None and pd.notna(minutes) and minutes >= 90:
                    keeper_games[player].append((shots, goals))
    return pd.DataFrame(rows)
