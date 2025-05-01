import json
import logging
import statistics
import math

from src.models import GTFPlayer, GTFTeam
from src.rating import aggregate_stats, update_player_rating
from src.storage import save_players_to_json, load_players_from_json, export_history
from src.analysis import calibrate_parameters, antifraud_smurf_detection

logger = logging.getLogger(__name__)

class GTFSystem:
    """Основной класс системы Glicko-Team-Flow для обновления рейтингов, сохранения и анализа."""

    def update_ratings(self, teams, ranks, stats=None, stat_weights=None, match_importance=1.0):
        """Обновляет рейтинги игроков всех команд для данного матча."""
        n = len(teams)
        if n == 0:
            logger.warning("update_ratings: нет команд для обновления.")
            return

        # Предвычисление средних значений по командам
        avg_mus = []
        mu_vars = []
        avg_phis = []
        avg_stats = []
        for team in teams:
            mus = [p.mu for p in team.players]
            avg_mus.append(statistics.mean(mus) if mus else 0)
            mu_vars.append(statistics.variance(mus) if len(mus) > 1 else 0)
            avg_phis.append(math.sqrt(statistics.mean([p.phi**2 for p in team.players])) if team.players else 0)
            team_stats = [aggregate_stats(p.stats, stat_weights) for p in team.players]
            avg_stats.append(statistics.mean(team_stats) if team_stats else 0)

        # Обновление каждого игрока
        for i, team in enumerate(teams):
            opp_avg_mu = statistics.mean([avg_mus[j] for j in range(n) if j != i]) if n > 1 else avg_mus[i]
            opp_mu_var = statistics.mean([mu_vars[j] for j in range(n) if j != i]) if n > 1 else mu_vars[i]
            opp_avg_phi = statistics.mean([avg_phis[j] for j in range(n) if j != i]) if n > 1 else avg_phis[i]
            opp_avg_stat = statistics.mean([avg_stats[j] for j in range(n) if j != i]) if n > 1 else avg_stats[i]
            # Результат команды i: 1.0 - rank/(n-1)
            outcome = 1.0 - (ranks[i] / (n - 1)) if n > 1 else 1.0
            for player in team.players:
                # Вызываем функцию обновления одного игрока
                update_player_rating(player,
                                     avg_stats[i], opp_avg_stat,
                                     opp_avg_mu, opp_avg_phi, opp_mu_var,
                                     outcome, match_importance, stat_weights)
                # Записываем историю
                player.history.append({'mu': player.mu, 'phi': player.phi, 'sigma': player.sigma})

    def save_players(self, players, path):
        """Сохраняет список игроков в файл JSON."""
        save_players_to_json(players, path)

    def load_players(self, path):
        """Загружает список игроков из JSON-файла."""
        return load_players_from_json(path)

    def export_history(self, players, path):
        """Экспортирует историю игроков в JSON-файл."""
        export_history(players, path)

    def calibrate(self, match_history):
        """Калибрует параметры системы по истории матчей и возвращает лучшие."""
        return calibrate_parameters(match_history)

    def antifraud_check(self, players):
        """Проверяет игроков на смурфинг (подозрительные роста рейтинга)."""
        return antifraud_smurf_detection(players) 