import json
import logging

from src.models import GTFPlayer

logger = logging.getLogger(__name__)


def save_players_to_json(players, path):
    """Сохраняет список игроков (GTFPlayer) в JSON-файл."""
    try:
        # Преобразуем объекты в словари
        player_dicts = [p.to_dict() if isinstance(p, GTFPlayer) else p for p in players]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(player_dicts, f, ensure_ascii=False, indent=2)
        logger.info(f'Players saved to {path}')
    except Exception as e:
        logger.error(f'Error saving players to {path}: {e}')


def load_players_from_json(path):
    """Загружает список игроков из JSON-файла и возвращает список GTFPlayer."""
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
    """Экспортирует историю всех игроков в JSON-файл {имя: история}."""
    try:
        history = {p.name: p.history for p in players}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f'History exported to {path}')
    except Exception as e:
        logger.error(f'Error exporting history to {path}: {e}') 