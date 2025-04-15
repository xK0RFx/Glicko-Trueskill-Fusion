import trueskill
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = 'trueskill_config.json'

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        logger.warning(f'Config file {path} not found, using defaults.')
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()
MU = config.get('MU', 25.0)
SIGMA = config.get('SIGMA', MU/3)
BETA = config.get('BETA', SIGMA/2)
TAU = config.get('TAU', SIGMA/100)
DRAW_PROB = config.get('DRAW_PROB', 0.1)

# Инициализация TrueSkill environment
TS = trueskill.TrueSkill(mu=MU, sigma=SIGMA, beta=BETA, tau=TAU, draw_probability=DRAW_PROB)

class Player:
    def __init__(self, name, rating=None, stat=0):
        self.name = name
        self.rating = rating if rating is not None else TS.create_rating()
        self.stat = stat
        self.history = []

    def get_rating(self):
        return self.rating.mu

    def get_sigma(self):
        return self.rating.sigma

    def get_confidence_interval(self, z=1.96):
        mu = self.rating.mu
        sigma = self.rating.sigma
        return (mu - z * sigma, mu + z * sigma)

    def to_dict(self):
        return {
            'name': self.name,
            'mu': self.rating.mu,
            'sigma': self.rating.sigma,
            'stat': self.stat,
            'history': self.history
        }

    @staticmethod
    def from_dict(d):
        p = Player(d['name'], trueskill.Rating(mu=d['mu'], sigma=d['sigma']), d.get('stat', 0))
        p.history = d.get('history', [])
        return p

class Team:
    def __init__(self, players):
        self.players = players

    def get_ratings(self):
        return [p.rating for p in self.players]

    def get_stats(self):
        return [p.stat for p in self.players]

    def to_dict(self):
        return [p.to_dict() for p in self.players]

    @staticmethod
    def from_dict(lst):
        return Team([Player.from_dict(d) for d in lst])

def update_ratings(teams, ranks, stats=None, stat_weight=0.0):
    """
    teams: список Team
    ranks: список int (0 - победитель, 1 - второй и т.д.)
    stats: список списков индивидуальных статов (или None)
    stat_weight: float, насколько сильно учитывать вклад статы (0 - не учитывать)
    """
    if stats is not None and stat_weight > 0:
        # Корректируем ranks с учётом индивидуального вклада
        avg_stats = [sum(team_stats)/len(team_stats) if team_stats else 0 for team_stats in stats]
        # Чем выше вклад, тем выше шанс победы (меньше rank)
        for i, team in enumerate(teams):
            for j, player in enumerate(team.players):
                if avg_stats[i] > 0:
                    adj = stat_weight * (player.stat - avg_stats[i]) / avg_stats[i]
                    # Корректируем рейтинг игрока через adjustment к score (эвристика)
                    player.rating = trueskill.Rating(mu=player.rating.mu + adj, sigma=player.rating.sigma)
    # Обновляем рейтинги через TrueSkill
    rating_groups = [team.get_ratings() for team in teams]
    new_ratings = TS.rate(rating_groups, ranks=ranks)
    for team, new_team_ratings in zip(teams, new_ratings):
        for player, new_rating in zip(team.players, new_team_ratings):
            player.rating = new_rating
            player.history.append({'mu': new_rating.mu, 'sigma': new_rating.sigma})

def save_players_to_json(players, path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in players], f, ensure_ascii=False, indent=2)
        logger.info(f'Players saved to {path}')
    except Exception as e:
        logger.error(f'Error saving players to {path}: {e}')

def load_players_from_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        players = [Player.from_dict(d) for d in data]
        logger.info(f'Loaded {len(players)} players from {path}')
        return players
    except Exception as e:
        logger.error(f'Error loading players from {path}: {e}')
        return [] 