import random
import time
from Allskill import GTFPlayer, GTFTeam, GTFSystem, aggregate_stats
import json
import visualization

def смоделировать_матч(команда_a, команда_b, system):
    результат = random.choices([1, 0, 0.5], weights=[0.45, 0.45, 0.1])[0]
    for p in команда_a:
        p.stats = {'kills': random.randint(0, 10)}
    for p in команда_b:
        p.stats = {'kills': random.randint(0, 10)}
    system.update_ratings([GTFTeam(команда_a), GTFTeam(команда_b)], [0, 1] if результат == 1 else ([1, 0] if результат == 0 else [0, 0]))
    return результат

def вывести_игроков(команда, метка):
    print(f"\n{метка}")
    for p in команда:
        ci = p.get_confidence_interval()
        print(f"{p.name}: Рейтинг={p.get_rating():.2f}, RD={p.get_rd():.2f}, ДИ=({ci[0]:.1f}, {ci[1]:.1f}), Волатильность={p.get_volatility():.4f}, Статы={p.stats}, Матчей={p.matches}")

def серия(команда_a, команда_b, n, system, история=None):
    for _ in range(n):
        смоделировать_матч(команда_a, команда_b, system)
        if история is not None:
            for p in команда_a + команда_b:
                if p.name not in история:
                    история[p.name] = {'рейтинг': [], 'rd': []}
                история[p.name]['рейтинг'].append(p.get_rating())
                история[p.name]['rd'].append(p.get_rd())

def тест_фаворит_аутсайдер(system):
    print("\n=== Явный фаворит против аутсайдера ===")
    команда_a = [GTFPlayer(f"Фаворит{i+1}", stats={'kills': 15}) for i in range(3)]
    команда_b = [GTFPlayer(f"Аутсайдер{i+1}", stats={'kills': 3}) for i in range(3)]
    история = {}
    серия(команда_a, команда_b, 100, system, история)
    вывести_игроков(команда_a, "Фавориты (после 100 матчей)")
    вывести_игроков(команда_b, "Аутсайдеры (после 100 матчей)")
    return история

def тест_влияние_статы(system):
    print("\n=== Влияние индивидуальной статистики ===")
    команда_a = [GTFPlayer(f"A{i+1}") for i in range(3)]
    команда_b = [GTFPlayer(f"B{i+1}") for i in range(3)]
    серия(команда_a, команда_b, 100, system)
    вывести_игроков(команда_a, "A (после 100 матчей)")
    вывести_игроков(команда_b, "B (после 100 матчей)")

def тест_история(system):
    print("\n=== Динамика рейтингов и RD ===")
    команда_a = [GTFPlayer(f"A{i+1}") for i in range(3)]
    команда_b = [GTFPlayer(f"B{i+1}") for i in range(3)]
    история = {}
    серия(команда_a, команда_b, 100, system, история)
    for имя, данные in история.items():
        print(f"{имя}: рейтинг[0]={данные['рейтинг'][0]:.1f}, рейтинг[-1]={данные['рейтинг'][-1]:.1f}, rd[0]={данные['rd'][0]:.1f}, rd[-1]={данные['rd'][-1]:.1f}")
    with open('history.json', 'w', encoding='utf-8') as f:
        json.dump(история, f, ensure_ascii=False, indent=2)
    print("История рейтингов сохранена в history.json")

def generate_player(name, role=None, smurf=False):
    stats = {
        'kills': random.randint(5, 20) if not smurf else random.randint(20, 40),
        'assists': random.randint(2, 10),
        'deaths': random.randint(0, 8) if not smurf else random.randint(0, 3),
        'mvp': random.randint(0, 2) if not smurf else random.randint(1, 4),
        'damage': random.randint(500, 2500) if not smurf else random.randint(2000, 4000)
    }
    return GTFPlayer(name, stats=stats, player_class=role)

def simulate_match(team_a, team_b, system, match_importance=1.0):
    skill_a = sum(aggregate_stats(p.stats) for p in team_a) + random.gauss(0, 10)
    skill_b = sum(aggregate_stats(p.stats) for p in team_b) + random.gauss(0, 10)
    if abs(skill_a - skill_b) < 5:
        score = 0.5
    elif skill_a > skill_b:
        score = 1.0
    else:
        score = 0.0
    system.update_ratings([GTFTeam(team_a), GTFTeam(team_b)], [0, 1] if score == 1.0 else ([1, 0] if score == 0.0 else [0, 0]), match_importance=match_importance)
    return score

def run_simulation():
    system = GTFSystem()
    players = [generate_player(f'Player{i+1}') for i in range(8)]
    smurfs = [generate_player(f'Smurf{i+1}', smurf=True) for i in range(2)]
    all_players = players + smurfs
    history = []
    for match_num in range(100):
        random.shuffle(all_players)
        team_a = all_players[:5]
        team_b = all_players[5:10]
        simulate_match(team_a, team_b, system, match_importance=random.choice([1.0, 2.0]))
        history.append([(p.name, p.get_rating()) for p in all_players])
    suspects = system.antifraud_check(all_players)
    print('Подозрительные смурфы:')
    for s in suspects:
        print(s)
    for p in all_players:
        visualization.plot_player_history(p)
        visualization.plot_player_stats(p)
    visualization.plot_multiple_players(all_players)
    visualization.plot_smurf_candidates(suspects)

def main():
    print("=== Базовый тест ===")
    system = GTFSystem()
    команда_a = [GTFPlayer(f"A{i+1}") for i in range(3)]
    команда_b = [GTFPlayer(f"B{i+1}") for i in range(3)]
    серия(команда_a, команда_b, 100, system)
    вывести_игроков(команда_a, "A (после 100 матчей)")
    вывести_игроков(команда_b, "B (после 100 матчей)")
    тест_фаворит_аутсайдер(system)
    тест_влияние_статы(system)
    тест_история(system)
    run_simulation()

if __name__ == "__main__":
    main() 