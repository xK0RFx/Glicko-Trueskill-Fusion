import math
import time
import statistics
import json
import logging

from src.config import (
    INITIAL_RATING_GLICKO, INITIAL_RD_GLICKO, INITIAL_VOLATILITY_GLICKO,
    ELO_OFFSET, SCALING_FACTOR, SECONDS_PER_DAY, TIME_DECAY_SIGMA_MULTIPLIER,
    logger # Use logger from config
)
# Forward declaration hint for type checkers, real import later if needed
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from .rating import aggregate_stats

class GTFPlayer:
    def __init__(self, name, rating=INITIAL_RATING_GLICKO, rd=INITIAL_RD_GLICKO, vol=INITIAL_VOLATILITY_GLICKO, stat=0, matches=0, last_match_time=None, player_class=None, stats=None):
        self.name = name
        # Convert Elo rating/RD to Glicko-2 scale
        self.mu = (rating - ELO_OFFSET) / SCALING_FACTOR
        self.phi = rd / SCALING_FACTOR
        self.sigma = vol # Volatility
        self.stat = stat # Deprecated? Aggregate stat kept for potential legacy use
        self.stats = stats if stats is not None else {} # Detailed stats dictionary
        self.matches = matches
        self.last_match_time = last_match_time if last_match_time is not None else time.time()
        self.player_class = player_class # Optional categorization
        self.history = [] # List to store rating history tuples/dicts

    def get_rating(self):
        """Returns the player's rating in the original Elo-like scale."""
        return self.mu * SCALING_FACTOR + ELO_OFFSET

    def get_rd(self):
        """Returns the player's rating deviation (RD) in the original Elo-like scale, after applying time decay."""
        self._pre_rating_rd_update()
        # Cap RD at the initial value
        return min(self.phi * SCALING_FACTOR, INITIAL_RD_GLICKO)

    def get_volatility(self):
        """Returns the player's rating volatility (sigma)."""
        return self.sigma

    def get_confidence_interval(self, z=1.96):
        """Returns the 95% confidence interval (lower, upper) for the rating."""
        rating = self.get_rating()
        rd = self.get_rd()
        return (rating - z * rd, rating + z * rd)

    def __repr__(self):
        """Provides a string representation of the player's current state."""
        rating = self.get_rating()
        rd = self.get_rd()
        vol = self.get_volatility()
        conf_interval = self.get_confidence_interval()
        # Use f-string for cleaner formatting
        return (
            f"{self.name}: Rating={rating:.2f} (±{rd:.1f}, 95% CI [{conf_interval[0]:.1f}, {conf_interval[1]:.1f}]), "
            f"Volatility={vol:.4f}, Matches={self.matches}, Stats={self.stats}"
            # Removed deprecated self.stat display
        )

    def _pre_rating_rd_update(self):
        """Internal method to update RD based on inactivity time before rating calculation."""
        time_now = time.time()
        time_diff_seconds = time_now - self.last_match_time
        days_inactive = time_diff_seconds / SECONDS_PER_DAY

        if days_inactive > 0:
            # Glicko-2 formula for RD increase over time
            increase_term = (self.sigma ** 2) * days_inactive * TIME_DECAY_SIGMA_MULTIPLIER
            try:
                phi_squared = self.phi**2 + increase_term
                # Ensure non-negative result before sqrt
                if phi_squared < 0: phi_squared = 0
                new_phi = math.sqrt(phi_squared)
            except (OverflowError, ValueError) as e:
                # Fallback to current phi if calculation fails
                new_phi = self.phi
                logger.warning(f'Could not calculate RD decay for player {self.name}: {e}')
            # Update phi, but cap it at the initial RD equivalent
            self.phi = min(new_phi, INITIAL_RD_GLICKO / SCALING_FACTOR)
            # Do NOT update last_match_time here, only after an actual match

    def to_dict(self):
        """Serializes the player object to a dictionary."""
        return {
            'name': self.name,
            'mu': self.mu,
            'phi': self.phi,
            'sigma': self.sigma,
            # 'stat': self.stat, # Keep if needed for loading old data
            'stats': self.stats,
            'matches': self.matches,
            'last_match_time': self.last_match_time,
            'player_class': self.player_class,
            'history': self.history
        }

    @staticmethod
    def from_dict(d):
        """Deserializes a player object from a dictionary."""
        # Handle potential loading of old data format without mu/phi/sigma
        if 'mu' in d and 'phi' in d and 'sigma' in d:
            rating = None # Will be derived from mu
            rd = None     # Will be derived from phi
            vol = d['sigma']
            mu = d['mu']
            phi = d['phi']
        else:
            # Fallback for older formats or direct Elo-like values
            rating = d.get('rating', INITIAL_RATING_GLICKO)
            rd = d.get('rd', INITIAL_RD_GLICKO)
            vol = d.get('sigma', INITIAL_VOLATILITY_GLICKO)
            mu = (rating - ELO_OFFSET) / SCALING_FACTOR
            phi = rd / SCALING_FACTOR

        p = GTFPlayer(
            name=d['name'],
            # Initialize directly if mu/phi provided, otherwise use rating/rd
            rating=rating if rating is not None else (mu * SCALING_FACTOR + ELO_OFFSET),
            rd=rd if rd is not None else (phi * SCALING_FACTOR),
            vol=vol,
            # stat=d.get('stat', 0), # Deprecated
            matches=d.get('matches', 0),
            last_match_time=d.get('last_match_time', None), # Keep None default if missing
            player_class=d.get('player_class', None),
            stats=d.get('stats', {}) # Ensure stats is always a dict
        )
        # Restore internal Glicko-2 parameters if they were saved
        p.mu = mu
        p.phi = phi
        p.history = d.get('history', []) # Ensure history is always a list
        return p

class GTFTeam:
    def __init__(self, players):
        # Ensure players is a list of GTFPlayer objects
        if not isinstance(players, list) or not all(isinstance(p, GTFPlayer) for p in players):
            raise TypeError("players must be a list of GTFPlayer objects")
        self.players = players

    def get_ratings(self):
        """Returns a list of Elo-like ratings for all players in the team."""
        return [p.get_rating() for p in self.players]

    def get_stats(self, stat_weights=None):
        """Returns a list of aggregated stats for all players in the team."""
        # Need the aggregate_stats function, assume it's imported or passed
        # For now, placeholder. Will be resolved when rating.py is created.
        # Replace with: from src.rating import aggregate_stats
        # return [aggregate_stats(p.stats, stat_weights) for p in self.players]
        # Temporary workaround until aggregate_stats is moved:
        temp_aggregate = lambda s, w: sum(s.values()) if w is None else sum(v * w.get(k, 0) for k, v in s.items())
        return [temp_aggregate(p.stats, stat_weights) for p in self.players]

    def to_dict(self):
        """Serializes the team (list of player dicts) to a list."""
        return [p.to_dict() for p in self.players]

    @staticmethod
    def from_dict(lst):
        """Deserializes a team from a list of player dictionaries."""
        if not isinstance(lst, list):
            raise TypeError("Team data must be a list of player dictionaries")
        return GTFTeam([GTFPlayer.from_dict(d) for d in lst]) 