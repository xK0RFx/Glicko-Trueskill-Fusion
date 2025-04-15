import math
import time # Понадобится для отслеживания времени последнего матча (опционально)
import statistics # Для расчета дисперсии

# --- Конфигурационные параметры Glicko-2 и системы ---
# Параметры Glicko-2
INITIAL_RATING_GLICKO = 1500 # Стандартное начальное значение в Glicko
INITIAL_RD_GLICKO = 350      # Стандартное начальное RD
INITIAL_VOLATILITY_GLICKO = 0.06 # Стандартная начальная волатильность
DEFAULT_TAU = 0.5       # Константа системы Glicko-2 (типичные значения 0.3-1.2)
                            # Влияет на то, как быстро растет волатильность
SCALING_FACTOR = 173.7178 # Коэффициент для перевода в/из шкалы Glicko-2
CONVERGENCE_TOLERANCE = 1e-6 # Точность для итеративного расчета волатильности

# Общие параметры
MIN_RD = 25             # Минимальное значение RD (чтобы неопределенность не исчезла)
# MIN_VOLATILITY = 0.01 # Минимальная волатильность (можно добавить)

# Конвертация в Elo-подобный рейтинг для отображения
ELO_OFFSET = 1500
ELO_SCALE = 400

# --- Новые параметры для гибридизации ---
TEAM_VARIANCE_WEIGHT = 0.1  # Коэффициент влияния дисперсии рейтинга команды противника на неопределенность матча (0 = выкл)
STAT_CONTRIBUTION_WEIGHT = 0.1 # Коэффициент влияния индивидуальной статы на изменение рейтинга (0 = выкл)
STAT_CONTRIBUTION_ALPHA = 0.5 # Параметр для расчета базового фактора вклада (как было раньше)

# --- Параметры для временного спада ---
SECONDS_PER_DAY = 60 * 60 * 24
# Коэффициент усиления sigma для расчета спада RD.
# В оригинале Glicko-2 это связано с кол-вом рейтинговых периодов.
# Здесь мы используем дни. Можно подбирать этот множитель.
TIME_DECAY_SIGMA_MULTIPLIER = 1.0

class Player:
    def __init__(self, name, rating=INITIAL_RATING_GLICKO, rd=INITIAL_RD_GLICKO, vol=INITIAL_VOLATILITY_GLICKO, stat=0, matches=0, last_match_time=None):
        self.name = name
        # Внутренние параметры Glicko-2
        self.mu = (rating - ELO_OFFSET) / SCALING_FACTOR # mu в шкале Glicko-2
        self.phi = rd / SCALING_FACTOR                  # phi (RD) в шкале Glicko-2
        self.sigma = vol                                # sigma (волатильность)
        self.stat = stat # Учитываем статистику
        # Дополнительная информация
        self.matches = matches
        # Если время не задано, используем текущее
        self.last_match_time = last_match_time if last_match_time is not None else time.time()

    def get_rating(self):
        """Возвращает рейтинг в привычной шкале (Elo-подобной)."""
        return self.mu * SCALING_FACTOR + ELO_OFFSET

    def get_rd(self):
        """Возвращает RD в привычной шкале."""
        # Применяем _pre_rating_rd_update ПЕРЕД возвратом RD, чтобы видеть актуальное значение
        self._pre_rating_rd_update()
        return min(self.phi * SCALING_FACTOR, INITIAL_RD_GLICKO)

    def get_volatility(self):
        """Возвращает волатильность."""
        return self.sigma

    # --- Добавляем метод доверительного интервала ---
    def get_confidence_interval(self, z=1.96):  # 95% доверительный интервал по умолчанию
        """Возвращает кортеж (нижняя_граница, верхняя_граница) доверительного интервала."""
        rating = self.get_rating() # Получаем актуальный рейтинг
        rd = self.get_rd()         # Получаем актуальный RD (с учетом спада)
        return (rating - z * rd, rating + z * rd)

    def __repr__(self):
        # Получаем значения через get_ методы, чтобы учесть _pre_rating_rd_update
        rating = self.get_rating()
        rd = self.get_rd() # RD здесь уже будет обновлен с учетом спада
        vol = self.get_volatility()
        # Посчитаем интервал для информации
        conf_interval = self.get_confidence_interval()
        return (f"{self.name}: Rating={rating:.2f} (±{rd:.1f}, 95% CI [{conf_interval[0]:.1f}, {conf_interval[1]:.1f}]), "
                f"Volatility={vol:.4f}, Stat={self.stat}, Matches={self.matches}")

    def _pre_rating_rd_update(self):
        """Обновление RD перед расчетом рейтинга (увеличение за неактивность)."""
        # Раскомментируем и используем формулу Glicko-2
        time_now = time.time()
        time_diff_seconds = time_now - self.last_match_time
        days_inactive = time_diff_seconds / SECONDS_PER_DAY

        if days_inactive > 0:
            # Формула увеличения RD: phi_new = sqrt(phi^2 + sigma^2 * t)
            # Где t - количество "рейтинговых периодов". Используем дни.
            increase_term = (self.sigma ** 2) * days_inactive * TIME_DECAY_SIGMA_MULTIPLIER
            try:
                phi_squared = self.phi**2 + increase_term
                # Проверка на отрицательное значение под корнем (маловероятно, но возможно из-за ошибок округления)
                if phi_squared < 0: phi_squared = 0
                new_phi = math.sqrt(phi_squared)
            except (OverflowError, ValueError):
                # В случае ошибки оставляем старое значение phi
                new_phi = self.phi
                # print(f"Warning: Could not calculate RD decay for player {self.name}")

            # Ограничиваем максимальный RD сверху стандартным начальным значением
            self.phi = min(new_phi, INITIAL_RD_GLICKO / SCALING_FACTOR)
            # Важно: НЕ обновляем self.last_match_time здесь,
            # это должно происходить ПОСЛЕ фактического матча

# --- Вспомогательные функции Glicko-2 ---

def _g(phi):
    """Вспомогательная функция g(phi)."""
    return 1 / math.sqrt(1 + 3 * (phi**2) / (math.pi**2))

def _E(mu, mu_opponent, phi_opponent):
    """Ожидаемый результат матча E."""
    return 1 / (1 + math.exp(-_g(phi_opponent) * (mu - mu_opponent)))

def _v(mu, mu_opponent, phi_opponent):
    """Ожидаемая дисперсия v."""
    g_val = _g(phi_opponent)
    E_val = _E(mu, mu_opponent, phi_opponent)
    # Добавим защиту от E_val близкого к 0 или 1, что может дать inf
    if E_val < 1e-9 or E_val > 1.0 - 1e-9:
        # В этом случае v стремится к бесконечности, что не очень хорошо.
        # Можно вернуть очень большое число или использовать аппроксимацию.
        # Или пересмотреть расчеты, если такое происходит часто.
        # Вернем просто большое значение, но это требует внимания.
        return 1e9
    return 1 / ((g_val**2) * E_val * (1 - E_val))

def _delta(mu, mu_opponent, phi_opponent, v, outcome):
    """Ожидаемое улучшение delta."""
    g_val = _g(phi_opponent)
    E_val = _E(mu, mu_opponent, phi_opponent)
    return v * g_val * (outcome - E_val)

def _compute_new_volatility(phi, v, delta, sigma, tau):
    """Итеративный расчет новой волатильности sigma'."""
    a = math.log(sigma**2)
    delta_sq = delta**2
    phi_sq = phi**2
    # Защита от v=0 или очень больших значений v, которые могут прийти из _v
    if v < 1e-9: v = 1e-9 # Избегаем деления на ноль в f(x)
    if v > 1e9: v = 1e9 # Ограничиваем сверху

    def f(x):
        exp_x = math.exp(x)
        denom = 2 * ((phi_sq + v + exp_x)**2)
        if abs(denom) < 1e-12: # Избегаем деления на ноль
             return -(x - a) / (tau**2) # Приблизительно, если знаменатель 0

        term1 = (exp_x * (delta_sq - phi_sq - v - exp_x)) / denom
        term2 = (x - a) / (tau**2)
        return term1 - term2

    # Итерационный алгоритм (Illinois modification of Regula Falsi)
    A = a
    if delta_sq > phi_sq + v:
        B = math.log(delta_sq - phi_sq - v)
    else:
        k = 1
        max_k = 20 # Ограничим число итераций поиска B
        while k <= max_k and f(a - k * tau) < 0:
            k += 1
        if k > max_k: # Не смогли найти B, возможно, проблема с параметрами
            # Возвращаем старую волатильность или минимально возможную?
            # Это проблемная ситуация, вернем старое значение.
             # print(f"Warning: Could not bound B in volatility calculation for phi={phi}, v={v}, delta={delta}, sigma={sigma}")
             return sigma
        B = a - k * tau

    fA = f(A)
    fB = f(B)
    side = 0

    max_iter = 100 # Ограничим число итераций решателя
    iter_count = 0
    while abs(B - A) > CONVERGENCE_TOLERANCE and iter_count < max_iter:
        C = A + (A - B) * fA / (fB - fA + 1e-12) # Добавим epsilon для стабильности
        fC = f(C)

        if fC * fB <= 0: # <= чтобы включить случай fC == 0
            A = B
            fA = fB
            B = C # Переместили B = C сюда
            fB = fC # Переместили fB = fC сюда
            # Illinois modification logic needs adjustment based on side
            if side == -1: fA /= 2
            side = 1 # Switched to B side
        else: # fC * fA > 0 or fC * fB > 0
            # Illinois modification
            fB = fB / 2 if side == 1 else fB # Reduce the slope on the side we keep
            A = C # Update A
            fA = fC # Update fA
            side = -1 # Switched to A side

        # B = C # B всегда становится C в конце итерации
        # fB = fC
        iter_count += 1

    if iter_count >= max_iter:
        # print(f"Warning: Volatility calculation did not converge within {max_iter} iterations.")
        # Вернем среднее между A и B или исходное sigma? Лучше исходное.
        return sigma

    return math.exp(A / 2) # Возвращаем sigma' = exp(A/2)

# --- Новая функция для учета статистики ---
def calculate_contribution_factor(player_stat, team_avg_stat):
    """Рассчитывает множитель вклада на основе статистики."""
    if abs(team_avg_stat) < 1e-6:
        # Если средняя стата близка к нулю, вклад трудно оценить.
        # Можно вернуть 1 (нейтрально) или 1 + alpha, если player_stat > 0.
        return 0 # Возвращаем 0, чтобы не влиять на delta в этом случае
    # Нормализуем отклонение, чтобы избежать слишком больших множителей
    relative_diff = (player_stat - team_avg_stat) / team_avg_stat
    # Применяем alpha и ограничиваем эффект, чтобы статистика не доминировала
    # Например, ограничиваем вклад диапазоном [-0.5, +0.5] * alpha
    contribution = max(-0.5, min(0.5, relative_diff)) * STAT_CONTRIBUTION_ALPHA
    return contribution

# --- Основная функция обновления ---

def update_player_rating(player, avg_team_stat, opponent_avg_mu, opponent_avg_phi, opponent_mu_variance, outcome):
    """Обновляет рейтинг одного игрока с учетом Glicko-2, дисперсии команды оппонента и статистики."""

    # Применяем спад RD ПЕРЕД всеми расчетами
    player._pre_rating_rd_update()

    # 1. Модификация phi оппонента
    variance_effect = opponent_mu_variance * (SCALING_FACTOR**2)
    # Добавим проверку на отрицательное значение под корнем
    if variance_effect < 0: variance_effect = 0
    normalized_variance_boost = math.sqrt(variance_effect) / (INITIAL_RATING_GLICKO + 1e-6) # epsilon
    effective_opponent_phi = opponent_avg_phi * (1 + TEAM_VARIANCE_WEIGHT * normalized_variance_boost)
    effective_opponent_phi = min(effective_opponent_phi, (INITIAL_RD_GLICKO * 1.5) / SCALING_FACTOR)

    # 2. Расчет стандартных величин
    g_opp = _g(effective_opponent_phi)
    E_val = _E(player.mu, opponent_avg_mu, effective_opponent_phi)
    v_val = _v(player.mu, opponent_avg_mu, effective_opponent_phi)
    delta_val = _delta(player.mu, opponent_avg_mu, effective_opponent_phi, v_val, outcome)

    # 3. Модификация delta статистикой
    contribution_factor = calculate_contribution_factor(player.stat, avg_team_stat)
    delta_multiplier = 1.0 + STAT_CONTRIBUTION_WEIGHT * contribution_factor
    # adjusted_delta = delta_val * delta_multiplier # Пока не используется напрямую

    # 4. Расчет новой волатильности
    new_sigma = _compute_new_volatility(player.phi, v_val, delta_val, player.sigma, DEFAULT_TAU)

    # 5. Обновляем RD (phi)
    phi_star = math.sqrt(player.phi**2 + new_sigma**2)
    v_inv = 1 / v_val if abs(v_val) > 1e-12 else 1e12
    new_phi = 1 / math.sqrt((1 / (phi_star**2 + 1e-12)) + v_inv)

    # 6. Обновляем рейтинг (mu), используя множитель от статистики
    update_term = new_phi**2 * g_opp * (outcome - E_val)
    final_update = update_term * delta_multiplier
    new_mu = player.mu + final_update

    # 7. Сохраняем новые значения
    player.mu = new_mu
    player.phi = max(new_phi, MIN_RD / SCALING_FACTOR)
    player.sigma = new_sigma
    player.matches += 1
    # Обновляем время последнего матча ПОСЛЕ всех расчетов
    player.last_match_time = time.time()

# --- Обновление рейтингов команд ---

def update_ratings(team_a, team_b, team_a_score):
    """Обновляет рейтинги всех игроков в командах по результату матча, используя гибридную систему."""
    if not team_a or not team_b:
        print("Warning: One or both teams are empty.")
        return # Нечего обновлять
    if team_a_score not in [0, 0.5, 1]:
        raise ValueError("team_a_score должен быть 0, 0.5 или 1")
    team_b_score = 1.0 - team_a_score

    # Расчет средних и дисперсий для команды A
    team_a_mus = [p.mu for p in team_a]
    avg_mu_a = statistics.mean(team_a_mus)
    # Дисперсия нужна, только если в команде больше 1 игрока
    mu_variance_a = statistics.variance(team_a_mus) if len(team_a_mus) > 1 else 0
    avg_phi_a = math.sqrt(sum(p.phi**2 for p in team_a) / len(team_a))
    avg_stat_a = statistics.mean(p.stat for p in team_a) if team_a else 0

    # Расчет средних и дисперсий для команды B
    team_b_mus = [p.mu for p in team_b]
    avg_mu_b = statistics.mean(team_b_mus)
    mu_variance_b = statistics.variance(team_b_mus) if len(team_b_mus) > 1 else 0
    avg_phi_b = math.sqrt(sum(p.phi**2 for p in team_b) / len(team_b))
    avg_stat_b = statistics.mean(p.stat for p in team_b) if team_b else 0

    # --- Сохраняем состояние перед обновлением (для передачи корректных opponent_avg) ---
    # Это важно, т.к. игроки обновляются по одному
    initial_state_a = [(p.mu, p.phi, p.sigma, p.stat) for p in team_a]
    initial_state_b = [(p.mu, p.phi, p.sigma, p.stat) for p in team_b]

    # Обновляем каждого игрока команды A (используя исходные средние команды B)
    for i, player in enumerate(team_a):
        # Используем avg_stat_a, рассчитанный по исходным статам команды А
        update_player_rating(player, avg_stat_a, avg_mu_b, avg_phi_b, mu_variance_b, team_a_score)

    # Обновляем каждого игрока команды B (используя исходные средние команды A)
    for i, player in enumerate(team_b):
        # Используем avg_stat_b, рассчитанный по исходным статам команды B
        update_player_rating(player, avg_stat_b, avg_mu_a, avg_phi_a, mu_variance_a, team_b_score)
