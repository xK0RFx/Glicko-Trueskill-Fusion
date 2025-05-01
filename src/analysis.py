import itertools
import statistics
import math
import logging

from src.models import GTFPlayer, GTFTeam
from src.config import set_calibration_params, logger
from src.system import GTFSystem


def antifraud_smurf_detection(players, min_matches=10, rating_growth_threshold=400, rd_threshold=80):
    """Обнаруживает игроков-смурфов на основе роста рейтинга и низкого RD."""
    suspects = []
    for p in players:
        if p.matches < min_matches:
            continue
        if not p.history or len(p.history) < min_matches:
            continue
        # Начальный рейтинг
        start_mu = p.history[0].get('mu', p.mu)
        start_rating = start_mu * p.history[0].get('sigma', 1)**0 + 0  # placeholder for offset
        # Фактический рейтинг
        end_rating = p.get_rating()
        growth = end_rating - start_rating
        rd = p.get_rd()
        if growth > rating_growth_threshold and rd < rd_threshold:
            suspects.append({
                'name': p.name,
                'growth': growth,
                'matches': p.matches,
                'rd': rd,
                'start_rating': start_rating,
                'end_rating': end_rating
            })
    return suspects


def calibrate_parameters(match_history, param_grid=None):
    """Перебирает сетку параметров, возвращает лучшие STAT, TEAM_VAR, TAU и RMSE."""
    if param_grid is None:
        param_grid = {
            'STAT_CONTRIBUTION_WEIGHT': [0.05, 0.1, 0.2, 0.3],
            'TEAM_VARIANCE_WEIGHT': [0.05, 0.1, 0.2],
            'DEFAULT_TAU': [0.3, 0.5, 0.7]
        }
    best_params = None
    best_rmse = float('inf')
    param_names = list(param_grid.keys())

    # Предвычисление матчей
    prep = []
    for match in match_history:
        teams = match['teams']
        ranks = match['ranks']
        stats = match.get('stats', None)
        importance = match.get('importance', 1.0)
        # Сериализация команд
        team_dicts = [[p.to_dict() for p in team] for team in teams]
        prep.append((team_dicts, ranks, stats, importance, match.get('real_result', 0)))

    # Перебор комбинаций
    for combo in itertools.product(*[param_grid[n] for n in param_names]):
        params = dict(zip(param_names, combo))
        set_calibration_params(params['STAT_CONTRIBUTION_WEIGHT'],
                               params['TEAM_VARIANCE_WEIGHT'],
                               params['DEFAULT_TAU'])
        system = GTFSystem()
        preds = []
        reals = []
        for team_dicts, ranks, stats, importance, real in prep:
            # Десериализация для обновления
            teams = [GTFTeam([GTFPlayer.from_dict(d) for d in team]) for team in team_dicts]
            system.update_ratings(teams, ranks, stats=stats, stat_weights=None, match_importance=importance)
            if len(teams) == 2:
                pred = teams[0].get_ratings()[0] - teams[1].get_ratings()[0]
            else:
                pred = 0
            preds.append(pred)
            reals.append(real)
        # RMSE
        mse = sum((p - r)**2 for p, r in zip(preds, reals)) / len(preds)
        rmse = math.sqrt(mse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params.copy()
    return best_params, best_rmse 