import random
import time
from Glicko_Trueskill_Fusion import Player, update_ratings
import json

def смоделировать_матч(команда_a, команда_b):
    результат = random.choices([1, 0, 0.5], weights=[0.45, 0.45, 0.1])[0]
    for p in команда_a:
        p.stat = random.randint(0, 10)
    for p in команда_b:
        p.stat = random.randint(0, 10)
    update_ratings(команда_a, команда_b, результат)
    return результат

def вывести_игроков(команда, метка):
    print(f"\n{метка}")
    for p in команда:
        ci = p.get_confidence_interval()
        print(f"{p.name}: Рейтинг={p.get_rating():.2f}, RD={p.get_rd():.2f}, ДИ=({ci[0]:.1f}, {ci[1]:.1f}), Волатильность={p.get_volatility():.4f}, Стата={p.stat}, Матчей={p.matches}")

def серия(команда_a, команда_b, n, история=None):
    for _ in range(n):
        смоделировать_матч(команда_a, команда_b)
        if история is not None:
            for p in команда_a + команда_b:
                if p.name not in история:
                    история[p.name] = {'рейтинг': [], 'rd': []}
                история[p.name]['рейтинг'].append(p.get_rating())
                история[p.name]['rd'].append(p.get_rd())

def тест_фаворит_аутсайдер():
    print("\n=== Явный фаворит против аутсайдера ===")
    команда_a = [Player(f"Фаворит{i+1}", rating=1700) for i in range(3)]
    команда_b = [Player(f"Аутсайдер{i+1}", rating=1300) for i in range(3)]
    история = {}
    серия(команда_a, команда_b, 100, история)
    вывести_игроков(команда_a, "Фавориты (после 100 матчей)")
    вывести_игроков(команда_b, "Аутсайдеры (после 100 матчей)")
    return история

def тест_влияние_статы():
    print("\n=== Влияние индивидуальной статистики ===")
    команда_a = [Player(f"A{i+1}") for i in range(3)]
    команда_b = [Player(f"B{i+1}") for i in range(3)]
    серия(команда_a, команда_b, 100)
    вывести_игроков(команда_a, "A (после 100 матчей)")
    вывести_игроков(команда_b, "B (после 100 матчей)")

def тест_история():
    print("\n=== Динамика рейтингов и RD ===")
    команда_a = [Player(f"A{i+1}") for i in range(3)]
    команда_b = [Player(f"B{i+1}") for i in range(3)]
    история = {}
    серия(команда_a, команда_b, 100, история)
    for имя, данные in история.items():
        print(f"{имя}: рейтинг[0]={данные['рейтинг'][0]:.1f}, рейтинг[-1]={данные['рейтинг'][-1]:.1f}, rd[0]={данные['rd'][0]:.1f}, rd[-1]={данные['rd'][-1]:.1f}")
    with open('history.json', 'w', encoding='utf-8') as f:
        json.dump(история, f, ensure_ascii=False, indent=2)
    print("История рейтингов сохранена в history.json")

def main():
    print("=== Базовый тест ===")
    команда_a = [Player(f"A{i+1}") for i in range(3)]
    команда_b = [Player(f"B{i+1}") for i in range(3)]
    серия(команда_a, команда_b, 100)
    вывести_игроков(команда_a, "A (после 100 матчей)")
    вывести_игроков(команда_b, "B (после 100 матчей)")
    тест_фаворит_аутсайдер()
    тест_влияние_статы()
    тест_история()

if __name__ == "__main__":
    main() 