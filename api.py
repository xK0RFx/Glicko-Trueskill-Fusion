from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from src.system import GTFSystem
from src.models import GTFPlayer
import os

system = GTFSystem()
app = FastAPI(title='Allskill API', description='Открытая рейтинговая система с мультистатами и антифродом', version='1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
PLAYERS_PATH = 'players.json'

class PlayerIn(BaseModel):
    name: str = Field(..., description='Имя игрока')
    stats: Optional[Dict[str, Any]] = Field(default=None, description='Мультистаты игрока')
    player_class: Optional[str] = Field(default=None, description='Класс или роль игрока')

class PlayerOut(BaseModel):
    name: str
    mu: float
    phi: float
    sigma: float
    matches: int
    last_match_time: float
    player_class: Optional[str]
    stats: Optional[Dict[str, Any]]
    history: list

class MatchIn(BaseModel):
    team_a: List[str] = Field(..., description='Список имён игроков команды A')
    team_b: List[str] = Field(..., description='Список имён игроков команды B')
    team_a_score: float = Field(..., description='Счёт команды A (1 — победа, 0 — поражение, 0.5 — ничья)')

async def get_all_players():
    if not os.path.exists(PLAYERS_PATH):
        return []
    return system.load_players(PLAYERS_PATH)

async def save_all_players(players):
    system.save_players(players, PLAYERS_PATH)

@app.get('/players', response_model=List[PlayerOut], summary='Получить всех игроков')
async def get_players():
    players = await get_all_players()
    return [p.to_dict() for p in players]

@app.get('/player/{name}', response_model=PlayerOut, summary='Получить игрока по имени')
async def get_player(name: str):
    players = await get_all_players()
    for p in players:
        if p.name == name:
            return p.to_dict()
    raise HTTPException(404, 'Игрок не найден')

@app.post('/player', response_model=PlayerOut, summary='Добавить нового игрока')
async def add_player(player: PlayerIn):
    players = await get_all_players()
    if any(p.name == player.name for p in players):
        raise HTTPException(400, 'Игрок с таким именем уже существует')
    new_player = GTFPlayer(player.name, stats=player.stats, player_class=player.player_class)
    players.append(new_player)
    await save_all_players(players)
    return new_player.to_dict()

@app.post('/match', summary='Обновить рейтинги после матча')
async def play_match(match: MatchIn):
    players = await get_all_players()
    team_a = [p for p in players if p.name in match.team_a]
    team_b = [p for p in players if p.name in match.team_b]
    if len(team_a) != len(match.team_a) or len(team_b) != len(match.team_b):
        raise HTTPException(400, 'Некорректные имена игроков')
    # Convert score to ranks
    rank_a = 1.0 - match.team_a_score
    rank_b = match.team_a_score
    system.update_ratings([team_a, team_b], [rank_a, rank_b])
    await save_all_players(players)
    return {'team_a': [p.to_dict() for p in team_a], 'team_b': [p.to_dict() for p in team_b]}

@app.get('/antifraud', summary='Проверить игроков на смурфинг/антифрод')
async def antifraud():
    players = await get_all_players()
    suspects = system.antifraud_check(players)
    return suspects 