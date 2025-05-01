import math
import time
import statistics
import logging

# Import models and config
from src.models import GTFPlayer # Assuming GTFPlayer is defined here
from src.config import (
    SCALING_FACTOR, INITIAL_RATING_GLICKO, INITIAL_RD_GLICKO,
    MIN_RD, STAT_CONTRIBUTION_ALPHA,
    logger,
    get_current_team_variance_weight, # Use getters for potentially calibrated values
    get_current_stat_contribution_weight
)
# Import Glicko-2 core functions
from src.glicko2 import _g, _E, _v, _delta, _compute_new_volatility

def aggregate_stats(stats, stat_weights=None):
    """Aggregates detailed player stats into a single score using weights."""
    if not stats:
        return 0
    # If no specific weights provided, simply sum all stat values
    if stat_weights is None:
        # Maybe warn or default to equal weights?
        # For now, replicates original behavior:
        return sum(stats.values())
    total = 0
    for k, v in stats.items():
        # Use weight 0 if stat not in weights
        w = stat_weights.get(k, 0)
        total += w * v
    return total

def calculate_contribution_factor(player_stat, team_avg_stat, opp_avg_stat=None):
    """Calculates a factor based on player stat relative to team and optionally opponent average."""
    # Avoid division by zero for team average stat
    if abs(team_avg_stat) < 1e-9: # Increased tolerance slightly
        contribution = 0
    else:
        try:
            # Player stat relative to their own team average
            relative_diff = (player_stat - team_avg_stat) / team_avg_stat
            # Cap the relative difference impact
            contribution = max(-0.5, min(0.5, relative_diff)) * STAT_CONTRIBUTION_ALPHA
        except ZeroDivisionError:
            contribution = 0

    # Optional: Add impact relative to opponent average stat
    if opp_avg_stat is not None and abs(opp_avg_stat) > 1e-9:
        try:
            opp_diff = (player_stat - opp_avg_stat) / opp_avg_stat
            # Add opponent difference effect, possibly capped or weighted differently
            # Original: contribution += 0.5 * STAT_CONTRIBUTION_ALPHA * opp_diff
            # Consider if capping opp_diff is also needed
            contribution += 0.5 * STAT_CONTRIBUTION_ALPHA * max(-1.0, min(1.0, opp_diff)) # Example capping
        except ZeroDivisionError:
            pass # Don't add opponent effect if their average is zero

    # Ensure final contribution is within a reasonable range, e.g., [-1, 1] * ALPHA ?
    # Current logic allows potentially larger values if both terms are large.
    # Revisit capping if needed based on desired behavior.
    return contribution

def update_player_rating(player, avg_team_stat, opp_avg_stat, opponent_avg_mu, opponent_avg_phi, opponent_mu_variance, outcome, match_importance=1.0, stat_weights=None):
    """Updates a single player's Glicko-2 parameters after a match."""
    if not isinstance(player, GTFPlayer):
        raise TypeError("player must be a GTFPlayer instance")

    player._pre_rating_rd_update() # Apply RD decay first

    # --- Calculate effective opponent parameters ---
    team_variance_weight = get_current_team_variance_weight()
    stat_contribution_weight = get_current_stat_contribution_weight()

    # Opponent team variance effect on their effective RD (phi)
    variance_effect = opponent_mu_variance * (SCALING_FACTOR**2)
    if variance_effect < 0: variance_effect = 0 # Should not happen but safety check
    # Normalize variance boost relative to initial rating range
    normalized_variance_boost = math.sqrt(variance_effect) / (INITIAL_RATING_GLICKO + 1e-6)
    # Increase opponent's effective phi based on their team's rating variance
    effective_opponent_phi = opponent_avg_phi * (1 + team_variance_weight * normalized_variance_boost)
    # Cap the effective phi to prevent extreme values (e.g., 1.5x initial RD)
    effective_opponent_phi = min(effective_opponent_phi, (INITIAL_RD_GLICKO * 1.5) / SCALING_FACTOR)
    # Ensure non-negative phi
    effective_opponent_phi = max(1e-9, effective_opponent_phi)

    # --- Core Glicko-2 Calculations ---
    g_opp = _g(effective_opponent_phi)
    E_val = _E(player.mu, opponent_avg_mu, effective_opponent_phi)
    v_val = _v(player.mu, opponent_avg_mu, effective_opponent_phi)
    delta_val = _delta(player.mu, opponent_avg_mu, effective_opponent_phi, v_val, outcome)

    # --- Calculate performance contribution factor ---
    player_agg_stat = aggregate_stats(player.stats, stat_weights)
    # Note: avg_team_stat and opp_avg_stat are pre-calculated and passed in
    contribution_factor = calculate_contribution_factor(player_agg_stat, avg_team_stat, opp_avg_stat)

    # Adjust the rating change (delta) based on performance and match importance
    # Ensure match_importance is positive
    match_importance = max(0, match_importance)
    delta_multiplier = 1.0 + stat_contribution_weight * contribution_factor
    # Cap the multiplier to prevent excessive boosts/penalties? e.g., [0.5, 1.5]?
    delta_multiplier = max(0.5, min(1.5, delta_multiplier)) # Example capping
    # Apply match importance AFTER calculating the base multiplier
    effective_delta_multiplier = 1 + (delta_multiplier - 1) * match_importance

    # --- Update Volatility (sigma) ---
    # Pass potentially calibrated tau via _compute_new_volatility's default mechanism
    new_sigma = _compute_new_volatility(player.phi, v_val, delta_val, player.sigma)

    # --- Update Rating Deviation (phi) ---
    try:
        phi_star_sq = player.phi**2 + new_sigma**2
        if phi_star_sq < 0: phi_star_sq = 1e-12 # Safety check
        phi_star = math.sqrt(phi_star_sq)
    except (OverflowError, ValueError) as e:
        logger.warning(f"Error calculating phi_star for {player.name}: {e}. Using current phi.")
        phi_star = player.phi

    # Inverse of v_val (add epsilon for stability if v_val is huge)
    v_inv = 1 / v_val if abs(v_val) > 1e-12 else 1e12
    try:
        # Denominator term for new_phi calculation
        new_phi_denom_sq = (1 / (phi_star**2 + 1e-12)) + v_inv # Add epsilon
        if new_phi_denom_sq < 1e-12: new_phi_denom_sq = 1e-12 # Avoid sqrt(0) or negative
        new_phi = 1 / math.sqrt(new_phi_denom_sq)
    except (OverflowError, ValueError) as e:
         logger.warning(f"Error calculating new_phi for {player.name}: {e}. Using current phi.")
         new_phi = player.phi

    # --- Update Rating (mu) ---
    # Original update term calculation
    update_term = new_phi**2 * g_opp * (outcome - E_val)
    # Apply the performance/importance multiplier
    final_update = update_term * effective_delta_multiplier
    new_mu = player.mu + final_update

    # --- Apply updates to player object ---
    player.mu = new_mu
    # Enforce minimum RD
    player.phi = max(new_phi, MIN_RD / SCALING_FACTOR)
    player.sigma = new_sigma
    player.matches += 1
    player.last_match_time = time.time()
    # Append history AFTER updates
    player.history.append({'mu': player.mu, 'phi': player.phi, 'sigma': player.sigma, 'time': player.last_match_time})

# --- Original top-level update_ratings function (for two teams) ---
# This might be deprecated or moved into GTFSystem later

def update_ratings_two_teams(team_a, team_b, team_a_score, stat_weights=None):
    """Updates ratings for two teams based on a single score (0, 0.5, 1)."""
    if not isinstance(team_a, list) or not all(isinstance(p, GTFPlayer) for p in team_a):
        raise TypeError("team_a must be a list of GTFPlayer objects")
    if not isinstance(team_b, list) or not all(isinstance(p, GTFPlayer) for p in team_b):
        raise TypeError("team_b must be a list of GTFPlayer objects")

    if not team_a or not team_b:
        logger.warning("update_ratings_two_teams: One or both teams are empty. Skipping update.")
        return
    if team_a_score not in [0, 0.5, 1]:
        logger.error(f"Invalid team_a_score: {team_a_score}. Must be 0, 0.5, or 1.")
        raise ValueError("team_a_score must be 0, 0.5, or 1")

    team_b_score = 1.0 - team_a_score

    # Calculate team averages needed for updates
    team_a_stats = [aggregate_stats(p.stats, stat_weights) for p in team_a]
    team_b_stats = [aggregate_stats(p.stats, stat_weights) for p in team_b]
    avg_stat_a = statistics.mean(team_a_stats) if team_a_stats else 0
    avg_stat_b = statistics.mean(team_b_stats) if team_b_stats else 0

    team_a_mus = [p.mu for p in team_a]
    avg_mu_a = statistics.mean(team_a_mus) if team_a_mus else 0
    mu_variance_a = statistics.variance(team_a_mus) if len(team_a_mus) > 1 else 0
    # Calculate average phi based on Glickman's multi-player suggestions (sqrt of mean squared phi)
    avg_phi_a = math.sqrt(statistics.mean([p.phi**2 for p in team_a])) if team_a else 0

    team_b_mus = [p.mu for p in team_b]
    avg_mu_b = statistics.mean(team_b_mus) if team_b_mus else 0
    mu_variance_b = statistics.variance(team_b_mus) if len(team_b_mus) > 1 else 0
    avg_phi_b = math.sqrt(statistics.mean([p.phi**2 for p in team_b])) if team_b else 0

    # Store initial state if needed for debugging (optional)
    # initial_state_a = [(p.name, p.mu, p.phi, p.sigma) for p in team_a]
    # initial_state_b = [(p.name, p.mu, p.phi, p.sigma) for p in team_b]

    # Update players in Team A (opponent is Team B)
    for player in team_a:
        update_player_rating(player, avg_stat_a, avg_stat_b, avg_mu_b, avg_phi_b, mu_variance_b, team_a_score, stat_weights=stat_weights)

    # Update players in Team B (opponent is Team A)
    for player in team_b:
        update_player_rating(player, avg_stat_b, avg_stat_a, avg_mu_a, avg_phi_a, mu_variance_a, team_b_score, stat_weights=stat_weights)

    # logger.debug(f"Updated ratings. Initial A: {initial_state_a}, Initial B: {initial_state_b}")
    # logger.debug(f"Updated ratings. Final A: {[(p.name, p.mu, p.phi, p.sigma) for p in team_a]}")
    # logger.debug(f"Updated ratings. Final B: {[(p.name, p.mu, p.phi, p.sigma) for p in team_b]}") 