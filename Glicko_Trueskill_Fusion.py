import math
import time
import statistics
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.json'

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        logger.warning(f'Config file {path} not found, using defaults.')
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()
INITIAL_RATING_GLICKO = config.get('INITIAL_RATING_GLICKO', 1500)
INITIAL_RD_GLICKO = config.get('INITIAL_RD_GLICKO', 350)
INITIAL_VOLATILITY_GLICKO = config.get('INITIAL_VOLATILITY_GLICKO', 0.06)
DEFAULT_TAU = config.get('DEFAULT_TAU', 0.5)
SCALING_FACTOR = config.get('SCALING_FACTOR', 173.7178)
CONVERGENCE_TOLERANCE = config.get('CONVERGENCE_TOLERANCE', 1e-6)
MIN_RD = config.get('MIN_RD', 25)
ELO_OFFSET = config.get('ELO_OFFSET', 1500)
TEAM_VARIANCE_WEIGHT = config.get('TEAM_VARIANCE_WEIGHT', 0.1)
STAT_CONTRIBUTION_WEIGHT = config.get('STAT_CONTRIBUTION_WEIGHT', 0.1)
STAT_CONTRIBUTION_ALPHA = config.get('STAT_CONTRIBUTION_ALPHA', 0.5)
SECONDS_PER_DAY = config.get('SECONDS_PER_DAY', 86400)
TIME_DECAY_SIGMA_MULTIPLIER = config.get('TIME_DECAY_SIGMA_MULTIPLIER', 1.0)

class Player:
    def __init__(self, name, rating=INITIAL_RATING_GLICKO, rd=INITIAL_RD_GLICKO, vol=INITIAL_VOLATILITY_GLICKO, stat=0, matches=0, last_match_time=None):
        self.name = name
        self.mu = (rating - ELO_OFFSET) / SCALING_FACTOR
        self.phi = rd / SCALING_FACTOR
        self.sigma = vol
        self.stat = stat
        self.matches = matches
        self.last_match_time = last_match_time if last_match_time is not None else time.time()

    def get_rating(self):
        return self.mu * SCALING_FACTOR + ELO_OFFSET

    def get_rd(self):
        self._pre_rating_rd_update()
        return min(self.phi * SCALING_FACTOR, INITIAL_RD_GLICKO)

    def get_volatility(self):
        return self.sigma

    def get_confidence_interval(self, z=1.96):
        rating = self.get_rating()
        rd = self.get_rd()
        return (rating - z * rd, rating + z * rd)

    def __repr__(self):
        rating = self.get_rating()
        rd = self.get_rd()
        vol = self.get_volatility()
        conf_interval = self.get_confidence_interval()
        return (f"{self.name}: Rating={rating:.2f} (±{rd:.1f}, 95% CI [{conf_interval[0]:.1f}, {conf_interval[1]:.1f}]), "
                f"Volatility={vol:.4f}, Stat={self.stat}, Matches={self.matches}")

    def _pre_rating_rd_update(self):
        time_now = time.time()
        time_diff_seconds = time_now - self.last_match_time
        days_inactive = time_diff_seconds / SECONDS_PER_DAY
        if days_inactive > 0:
            increase_term = (self.sigma ** 2) * days_inactive * TIME_DECAY_SIGMA_MULTIPLIER
            try:
                phi_squared = self.phi**2 + increase_term
                if phi_squared < 0: phi_squared = 0
                new_phi = math.sqrt(phi_squared)
            except (OverflowError, ValueError):
                new_phi = self.phi
                logger.warning(f'Could not calculate RD decay for player {self.name}')
            self.phi = min(new_phi, INITIAL_RD_GLICKO / SCALING_FACTOR)

    def to_dict(self):
        return {
            'name': self.name,
            'mu': self.mu,
            'phi': self.phi,
            'sigma': self.sigma,
            'stat': self.stat,
            'matches': self.matches,
            'last_match_time': self.last_match_time
        }

    @staticmethod
    def from_dict(d):
        return Player(
            name=d['name'],
            rating=d.get('mu', 0) * SCALING_FACTOR + ELO_OFFSET if 'mu' in d else d.get('rating', INITIAL_RATING_GLICKO),
            rd=d.get('phi', 2.014) * SCALING_FACTOR if 'phi' in d else d.get('rd', INITIAL_RD_GLICKO),
            vol=d.get('sigma', INITIAL_VOLATILITY_GLICKO),
            stat=d.get('stat', 0),
            matches=d.get('matches', 0),
            last_match_time=d.get('last_match_time', None)
        )

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

def _g(phi):
    return 1 / math.sqrt(1 + 3 * (phi**2) / (math.pi**2))

def _E(mu, mu_opponent, phi_opponent):
    return 1 / (1 + math.exp(-_g(phi_opponent) * (mu - mu_opponent)))

def _v(mu, mu_opponent, phi_opponent):
    g_val = _g(phi_opponent)
    E_val = _E(mu, mu_opponent, phi_opponent)
    if E_val < 1e-9 or E_val > 1.0 - 1e-9:
        return 1e9
    return 1 / ((g_val**2) * E_val * (1 - E_val) + 1e-12)

def _delta(mu, mu_opponent, phi_opponent, v, outcome):
    g_val = _g(phi_opponent)
    E_val = _E(mu, mu_opponent, phi_opponent)
    v_capped = min(v, 1e9)
    return v_capped * g_val * (outcome - E_val)

def _compute_new_volatility(phi, v, delta, sigma, tau):
    a = math.log(sigma**2)
    delta_sq = delta**2
    phi_sq = phi**2
    if v < 1e-9: v = 1e-9
    if v > 1e9: v = 1e9
    def f(x):
        exp_x = math.exp(x)
        denom = 2 * ((phi_sq + v + exp_x)**2)
        if abs(denom) < 1e-12:
             return -(x - a) / (tau**2 + 1e-12)
        term1 = (exp_x * (delta_sq - phi_sq - v - exp_x)) / denom
        term2 = (x - a) / (tau**2 + 1e-12)
        return term1 - term2
    A = a
    if delta_sq > phi_sq + v:
        try:
            B = math.log(delta_sq - phi_sq - v)
        except ValueError:
             return sigma
    else:
        k = 1
        max_k = 20
        while k <= max_k:
            try:
                if f(a - k * tau) < 0:
                   break
            except (OverflowError, ValueError):
                k = max_k + 1
                break
            k += 1
        if k > max_k:
             return sigma
        B = a - k * tau
    try:
        fA = f(A)
        fB = f(B)
    except (OverflowError, ValueError):
         return sigma
    side = 0
    max_iter = 100
    iter_count = 0
    while abs(B - A) > CONVERGENCE_TOLERANCE and iter_count < max_iter:
        try:
            denom_regula = fB - fA
            if abs(denom_regula) < 1e-12: denom_regula = 1e-12 * math.copysign(1, denom_regula)
            C = A + (A - B) * fA / denom_regula
            fC = f(C)
        except (OverflowError, ValueError):
             break
        if fC * fB <= 0:
            A = B
            fA = fB
            if side == -1: fA /= 2
            side = 1
        else:
            if side == 1: fB /= 2
            side = -1
        B = C
        fB = fC
        iter_count += 1
    if iter_count >= max_iter:
        return sigma
    return math.exp(A / 2)

def calculate_contribution_factor(player_stat, team_avg_stat):
    if abs(team_avg_stat) < 1e-6:
        return 0
    try:
      relative_diff = (player_stat - team_avg_stat) / team_avg_stat
    except ZeroDivisionError:
      return 0
    contribution = max(-0.5, min(0.5, relative_diff)) * STAT_CONTRIBUTION_ALPHA
    return contribution

def update_player_rating(player, avg_team_stat, opponent_avg_mu, opponent_avg_phi, opponent_mu_variance, outcome):
    player._pre_rating_rd_update()
    variance_effect = opponent_mu_variance * (SCALING_FACTOR**2)
    if variance_effect < 0: variance_effect = 0
    normalized_variance_boost = math.sqrt(variance_effect) / (INITIAL_RATING_GLICKO + 1e-6)
    effective_opponent_phi = opponent_avg_phi * (1 + TEAM_VARIANCE_WEIGHT * normalized_variance_boost)
    effective_opponent_phi = min(effective_opponent_phi, (INITIAL_RD_GLICKO * 1.5) / SCALING_FACTOR)
    g_opp = _g(effective_opponent_phi)
    E_val = _E(player.mu, opponent_avg_mu, effective_opponent_phi)
    v_val = _v(player.mu, opponent_avg_mu, effective_opponent_phi)
    delta_val = _delta(player.mu, opponent_avg_mu, effective_opponent_phi, v_val, outcome)
    contribution_factor = calculate_contribution_factor(player.stat, avg_team_stat)
    delta_multiplier = 1.0 + STAT_CONTRIBUTION_WEIGHT * contribution_factor
    new_sigma = _compute_new_volatility(player.phi, v_val, delta_val, player.sigma, DEFAULT_TAU)
    phi_star = math.sqrt(player.phi**2 + new_sigma**2)
    v_inv = 1 / v_val if abs(v_val) > 1e-12 else 1e12
    new_phi = 1 / math.sqrt((1 / (phi_star**2 + 1e-12)) + v_inv)
    update_term = new_phi**2 * g_opp * (outcome - E_val)
    final_update = update_term * delta_multiplier
    new_mu = player.mu + final_update
    player.mu = new_mu
    player.phi = max(new_phi, MIN_RD / SCALING_FACTOR)
    player.sigma = new_sigma
    player.matches += 1
    player.last_match_time = time.time()

def update_ratings(team_a, team_b, team_a_score):
    if not team_a or not team_b:
        logger.warning("One or both teams are empty.")
        return
    if team_a_score not in [0, 0.5, 1]:
        logger.error("team_a_score должен быть 0, 0.5 или 1")
        raise ValueError("team_a_score должен быть 0, 0.5 или 1")
    team_b_score = 1.0 - team_a_score
    team_a_mus = [p.mu for p in team_a]
    avg_mu_a = statistics.mean(team_a_mus) if team_a_mus else 0
    mu_variance_a = statistics.variance(team_a_mus) if len(team_a_mus) > 1 else 0
    avg_phi_a = math.sqrt(sum(p.phi**2 for p in team_a) / len(team_a)) if team_a else 0
    avg_stat_a = statistics.mean(p.stat for p in team_a) if team_a else 0
    team_b_mus = [p.mu for p in team_b]
    avg_mu_b = statistics.mean(team_b_mus) if team_b_mus else 0
    mu_variance_b = statistics.variance(team_b_mus) if len(team_b_mus) > 1 else 0
    avg_phi_b = math.sqrt(sum(p.phi**2 for p in team_b) / len(team_b)) if team_b else 0
    avg_stat_b = statistics.mean(p.stat for p in team_b) if team_b else 0
    initial_state_a = [(p.mu, p.phi, p.sigma, p.stat) for p in team_a]
    initial_state_b = [(p.mu, p.phi, p.sigma, p.stat) for p in team_b]
    for i, player in enumerate(team_a):
        update_player_rating(player, avg_stat_a, avg_mu_b, avg_phi_b, mu_variance_b, team_a_score)
    for i, player in enumerate(team_b):
        update_player_rating(player, avg_stat_b, avg_mu_a, avg_phi_a, mu_variance_a, team_b_score)
