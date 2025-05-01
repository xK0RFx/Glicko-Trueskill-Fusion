import json
import logging

from src.models import GTFPlayer

logger = logging.getLogger(__name__)


def save_players_to_json(players, path):
    """Save players (GTFPlayer) list to a JSON file."""
    try:
        # Convert player objects to dictionaries
        player_dicts = [p.to_dict() if isinstance(p, GTFPlayer) else p for p in players]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(player_dicts, f, ensure_ascii=False, indent=2)
        logger.info(f'Players saved to {path}')
    except Exception as e:
        logger.error(f'Error saving players to {path}: {e}')


def load_players_from_json(path):
    """Load players from a JSON file and return a list of GTFPlayer."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        players = [GTFPlayer.from_dict(d) for d in data]
        logger.info(f'Loaded {len(players)} players from {path}')
        return players
    except FileNotFoundError:
        logger.warning(f'Player file {path} not found. Returning empty list.')
        return []
    except Exception as e:
        logger.error(f'Error loading players from {path}: {e}')
        return []


def export_history(players, path):
    """Export history of all players to a JSON file mapping name to history."""
    try:
        history = {p.name: p.history for p in players}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f'History exported to {path}')
    except Exception as e:
        logger.error(f'Error exporting history to {path}: {e}') 