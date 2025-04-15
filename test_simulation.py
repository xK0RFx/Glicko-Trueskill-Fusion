import random
import time
from Glicko_Trueskill_Fusion import Player, update_ratings

def simulate_match(team_a, team_b):
    # Случайно определяем результат: 1 - победа A, 0 - победа B, 0.5 - ничья
    result = random.choices([1, 0, 0.5], weights=[0.45, 0.45, 0.1])[0]
    # Генерируем случайную статистику для игроков
    for p in team_a:
        p.stat = random.randint(0, 10)
    for p in team_b:
        p.stat = random.randint(0, 10)
    update_ratings(team_a, team_b, result)
    return result

def print_players(team, label):
    print(f"\n{label}")
    for p in team:
        ci = p.get_confidence_interval()
        print(f"{p.name}: Rating={p.get_rating():.2f}, RD={p.get_rd():.2f}, CI=({ci[0]:.1f}, {ci[1]:.1f}), Vol={p.get_volatility():.4f}, Stat={p.stat}, Matches={p.matches}")

def run_series(team_a, team_b, n_matches, history=None):
    for i in range(n_matches):
        simulate_match(team_a, team_b)
        if history is not None:
            for p in team_a + team_b:
                if p.name not in history:
                    history[p.name] = {'rating': [], 'rd': []}
                history[p.name]['rating'].append(p.get_rating())
                history[p.name]['rd'].append(p.get_rd())

def test_favorite_vs_underdog():
    print("\n=== Тест: Явный фаворит против аутсайдера ===")
    team_a = [Player(f"Fav{i+1}", rating=1700) for i in range(3)]
    team_b = [Player(f"Dog{i+1}", rating=1300) for i in range(3)]
    history = {}
    run_series(team_a, team_b, 100, history)
    print_players(team_a, "Fav Team (после 100 матчей)")
    print_players(team_b, "Dog Team (после 100 матчей)")
    return history

def test_stat_contribution(weight):
    print(f"\n=== Тест: Влияние статистики (STAT_CONTRIBUTION_WEIGHT={weight}) ===")
    from Glicko_Trueskill_Fusion import STAT_CONTRIBUTION_WEIGHT
    import Glicko_Trueskill_Fusion as fusion
    fusion.STAT_CONTRIBUTION_WEIGHT = weight
    team_a = [Player(f"A{i+1}") for i in range(3)]
    team_b = [Player(f"B{i+1}") for i in range(3)]
    run_series(team_a, team_b, 100)
    print_players(team_a, f"Team A (STAT_CONTRIBUTION_WEIGHT={weight})")
    print_players(team_b, f"Team B (STAT_CONTRIBUTION_WEIGHT={weight})")

def test_history_graph():
    print("\n=== Тест: Динамика рейтингов и RD ===")
    team_a = [Player(f"A{i+1}") for i in range(3)]
    team_b = [Player(f"B{i+1}") for i in range(3)]
    history = {}
    run_series(team_a, team_b, 100, history)
    for name, data in history.items():
        print(f"{name}: rating[0]={data['rating'][0]:.1f}, rating[-1]={data['rating'][-1]:.1f}, rd[0]={data['rd'][0]:.1f}, rd[-1]={data['rd'][-1]:.1f}")
    # Можно сохранить history в json для построения графика
    import json
    with open('history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print("История рейтингов сохранена в history.json")

def main():
    # Обычный тест
    print("=== Базовый тест ===")
    team_a = [Player(f"A{i+1}") for i in range(3)]
    team_b = [Player(f"B{i+1}") for i in range(3)]
    run_series(team_a, team_b, 100)
    print_players(team_a, "Team A (после 100 матчей)")
    print_players(team_b, "Team B (после 100 матчей)")
    # Тест фаворита
    test_favorite_vs_underdog()
    # Тест с разным влиянием статистики
    test_stat_contribution(0.0)
    test_stat_contribution(0.5)
    test_stat_contribution(1.0)
    # Тест истории для графика
    test_history_graph()

if __name__ == "__main__":
    main() 