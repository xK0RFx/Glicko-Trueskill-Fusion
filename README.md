# Allskill

Гибридная рейтинговая система для командных и индивидуальных игр с поддержкой мультистатов, временного спада неопределённости и антифрода.

## Возможности

- Glicko-2 с расширениями (TrueSkill-like, мультистаты, decay, доверительные интервалы)
- Гибкая настройка через config.json
- Поддержка командных матчей, ролей, индивидуальных статов
- Автоматическая калибровка параметров
- Поиск смурфов и антифрода
- Сохранение/загрузка игроков в JSON

## Быстрый старт

1. Установи Python 3.8+
2. Склонируй репозиторий и перейди в папку проекта
3. Настрой параметры в `config.json` (опционально)
4. Импортируй и используй:

```python
from Allskill import GTFPlayer, GTFTeam, update_ratings, save_players_to_json, load_players_from_json, GTFSystem

# Создание игроков
alice = GTFPlayer('Alice', stats={'kills': 10, 'assists': 5})
bob = GTFPlayer('Bob', stats={'kills': 7, 'assists': 8})
team_a = [alice]
team_b = [bob]

# Обновление рейтинга после матча (Alice победила)
update_ratings(team_a, team_b, team_a_score=1)

# Сохранение
save_players_to_json([alice, bob], 'players.json')

# Загрузка
players = load_players_from_json('players.json')
```

## Основные классы и функции

- `GTFPlayer` — игрок (рейтинг, RD, волатильность, мультистаты, история)
- `GTFTeam` — команда игроков
- `update_ratings(team_a, team_b, team_a_score)` — обновление рейтингов двух команд
- `save_players_to_json(players, path)` / `load_players_from_json(path)` — работа с файлами
- `GTFSystem` — универсальный класс для мультикомандных матчей, калибровки и антифрода
- `get_confidence_interval()` — доверительный интервал рейтинга игрока
- `antifraud_smurf_detection(players)` — поиск подозрительных аккаунтов

## Формат игрока

```json
{
	"name": "Alice",
	"mu": 0.0,
	"phi": 2.014,
	"sigma": 0.06,
	"matches": 5,
	"last_match_time": 1710000000.0,
	"player_class": "support",
	"stats": { "kills": 10, "assists": 5 },
	"history": []
}
```

## Конфигурация (config.json)

- `INITIAL_RATING_GLICKO` — стартовый рейтинг
- `INITIAL_RD_GLICKO` — стартовая неопределённость
- `INITIAL_VOLATILITY_GLICKO` — стартовая волатильность
- `DEFAULT_TAU` — параметр tau
- `SCALING_FACTOR` — масштаб рейтинга
- `CONVERGENCE_TOLERANCE` — точность расчёта
- `MIN_RD` — минимальный RD
- `ELO_OFFSET` — смещение шкалы
- `TEAM_VARIANCE_WEIGHT` — вес дисперсии команды
- `STAT_CONTRIBUTION_WEIGHT` — вес мультистатов
- `STAT_CONTRIBUTION_ALPHA` — коэффициент вклада стата
- `SECONDS_PER_DAY` — секунд в дне
- `TIME_DECAY_SIGMA_MULTIPLIER` — спад RD

## Лицензия

MIT © K0RF
