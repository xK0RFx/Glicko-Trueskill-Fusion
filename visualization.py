import matplotlib.pyplot as plt

SCALING_FACTOR = 173.7178
ELO_OFFSET = 1500

# Визуализация истории рейтинга и доверительного интервала

def plot_player_history(player):
    ratings = [h['mu'] * SCALING_FACTOR + ELO_OFFSET for h in player.history]
    rds = [h['phi'] * SCALING_FACTOR for h in player.history]
    matches = list(range(1, len(ratings) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(matches, ratings, label='Рейтинг')
    plt.fill_between(matches, [r - 2*rd for r, rd in zip(ratings, rds)],
                     [r + 2*rd for r, rd in zip(ratings, rds)], color='gray', alpha=0.2, label='95% ДИ')
    plt.title(f'Динамика рейтинга: {player.name}')
    plt.xlabel('Матч')
    plt.ylabel('Рейтинг')
    plt.legend()
    plt.grid(True)
    plt.show()

# Визуализация мультистатов игрока (последние значения)
def plot_player_stats(player):
    if not hasattr(player, 'stats') or not player.stats:
        print('Нет данных для визуализации')
        return
    stats_keys = list(player.stats.keys())
    values = [player.stats.get(k, 0) for k in stats_keys]
    plt.figure(figsize=(8, 4))
    plt.bar(stats_keys, values)
    plt.title(f'Статы игрока: {player.name}')
    plt.ylabel('Значение')
    plt.show()

# Сравнение истории рейтинга нескольких игроков
def plot_multiple_players(players):
    plt.figure(figsize=(10, 5))
    for player in players:
        ratings = [h['mu'] * SCALING_FACTOR + ELO_OFFSET for h in player.history]
        matches = list(range(1, len(ratings) + 1))
        plt.plot(matches, ratings, label=player.name)
    plt.title('Сравнение рейтингов игроков')
    plt.xlabel('Матч')
    plt.ylabel('Рейтинг')
    plt.legend()
    plt.grid(True)
    plt.show()

# Визуализация роста рейтинга для подозрительных игроков (смурфов)
def plot_smurf_candidates(suspects):
    if not suspects:
        print('Нет подозрительных игроков')
        return
    names = [s['name'] for s in suspects]
    growth = [s['growth'] for s in suspects]
    plt.figure(figsize=(8, 4))
    plt.bar(names, growth, color='red')
    plt.title('Возможные смурфы: рост рейтинга')
    plt.ylabel('Рост рейтинга')
    plt.xlabel('Игрок')
    plt.show()

# Пример использования:
# from Allskill import GTFPlayer, GTFSystem
# import visualization
# player = ...
# visualization.plot_player_history(player)
# visualization.plot_player_stats(player)
# visualization.plot_multiple_players([player1, player2, player3])
# suspects = GTFSystem().antifraud_check(players)
# visualization.plot_smurf_candidates(suspects) 