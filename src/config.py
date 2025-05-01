import json
import logging
import os
import time # Needed for default last_match_time, though maybe move later

logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.json'

def load_config(path=CONFIG_PATH):
    defaults = {
        'INITIAL_RATING_GLICKO': 1500,
        'INITIAL_RD_GLICKO': 350,
        'INITIAL_VOLATILITY_GLICKO': 0.06,
        'DEFAULT_TAU': 0.5,
        'SCALING_FACTOR': 173.7178,
        'CONVERGENCE_TOLERANCE': 1e-6,
        'MIN_RD': 25,
        'ELO_OFFSET': 1500,
        'TEAM_VARIANCE_WEIGHT': 0.1,
        'STAT_CONTRIBUTION_WEIGHT': 0.1,
        'STAT_CONTRIBUTION_ALPHA': 0.5,
        'SECONDS_PER_DAY': 86400,
        'TIME_DECAY_SIGMA_MULTIPLIER': 1.0
    }
    if not os.path.exists(path):
        logger.warning(f'Config file {path} not found, using defaults.')
        return defaults
    try:
        with open(path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        # Merge loaded config with defaults, ensuring all keys exist
        config = defaults.copy()
        config.update(loaded_config)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {path}: {e}. Using defaults.")
        return defaults

config = load_config()

# Extract constants from config for easier access
INITIAL_RATING_GLICKO = config['INITIAL_RATING_GLICKO']
INITIAL_RD_GLICKO = config['INITIAL_RD_GLICKO']
INITIAL_VOLATILITY_GLICKO = config['INITIAL_VOLATILITY_GLICKO']
DEFAULT_TAU = config['DEFAULT_TAU']
SCALING_FACTOR = config['SCALING_FACTOR']
CONVERGENCE_TOLERANCE = config['CONVERGENCE_TOLERANCE']
MIN_RD = config['MIN_RD']
ELO_OFFSET = config['ELO_OFFSET']
TEAM_VARIANCE_WEIGHT = config['TEAM_VARIANCE_WEIGHT']
STAT_CONTRIBUTION_WEIGHT = config['STAT_CONTRIBUTION_WEIGHT']
STAT_CONTRIBUTION_ALPHA = config['STAT_CONTRIBUTION_ALPHA']
SECONDS_PER_DAY = config['SECONDS_PER_DAY']
TIME_DECAY_SIGMA_MULTIPLIER = config['TIME_DECAY_SIGMA_MULTIPLIER']

# Global mutable parameters (used by calibration, might need rethinking)
# We define them here initially from the loaded config
# Consider passing these explicitly or using a class instance instead of globals
CURRENT_STAT_CONTRIBUTION_WEIGHT = STAT_CONTRIBUTION_WEIGHT
CURRENT_TEAM_VARIANCE_WEIGHT = TEAM_VARIANCE_WEIGHT
CURRENT_DEFAULT_TAU = DEFAULT_TAU

def set_calibration_params(stat_weight, team_var_weight, tau):
    """
    Sets the global parameters used during calibration.
    """
    global CURRENT_STAT_CONTRIBUTION_WEIGHT, CURRENT_TEAM_VARIANCE_WEIGHT, CURRENT_DEFAULT_TAU
    CURRENT_STAT_CONTRIBUTION_WEIGHT = stat_weight
    CURRENT_TEAM_VARIANCE_WEIGHT = team_var_weight
    CURRENT_DEFAULT_TAU = tau

def get_current_tau():
    """
    Gets the current tau value (potentially modified by calibration).
    """
    return CURRENT_DEFAULT_TAU

def get_current_team_variance_weight():
    """
    Gets the current team variance weight (potentially modified by calibration).
    """
    return CURRENT_TEAM_VARIANCE_WEIGHT

def get_current_stat_contribution_weight():
    """
    Gets the current stat contribution weight (potentially modified by calibration).
    """
    return CURRENT_STAT_CONTRIBUTION_WEIGHT

# Consider moving logger setup elsewhere (e.g., main entry point)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s') 