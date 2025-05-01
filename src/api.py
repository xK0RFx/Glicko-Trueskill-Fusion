from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from src.system import GTFSystem
from src.models import GTFPlayer
import os

# Initialize system and FastAPI app
system = GTFSystem()
app = FastAPI(
    title='Allskill API',
    description='Open rating system with multistats and anti-fraud features',
    version='1.0'
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

PLAYERS_PATH = 'players.json'

class PlayerIn(BaseModel):
    name: str = Field(..., description='Player name')
    stats: Optional[Dict[str, Any]] = Field(None, description='Player multistats')
    player_class: Optional[str] = Field(None, description='Player class or role')

class PlayerOut(BaseModel):
    name: str
    mu: float
    phi: float
    sigma: float
    matches: int
    last_match_time: float
    player_class: Optional[str]
    stats: Optional[Dict[str, Any]]
    history: List[Dict[str, Any]]

class MatchIn(BaseModel):
    team_a: List[str] = Field(..., description='Player names for team A')
    team_b: List[str] = Field(..., description='Player names for team B')
    team_a_score: float = Field(..., description='Score for team A (1.0 win, 0.5 draw, 0.0 loss)')

async def get_all_players() -> List[GTFPlayer]:
    if not os.path.exists(PLAYERS_PATH):
        return []
    return system.load_players(PLAYERS_PATH)

async def save_all_players(players: List[GTFPlayer]) -> None:
    system.save_players(players, PLAYERS_PATH)

@app.get('/players', response_model=List[PlayerOut], summary='Get all players')
async def get_players() -> List[Dict[str, Any]]:
    players = await get_all_players()
    return [p.to_dict() for p in players]

@app.get('/player/{name}', response_model=PlayerOut, summary='Get player by name')
async def get_player(name: str) -> Dict[str, Any]:
    players = await get_all_players()
    for p in players:
        if p.name == name:
            return p.to_dict()
    raise HTTPException(status_code=404, detail='Player not found')

@app.post('/player', response_model=PlayerOut, summary='Add a new player')
async def add_player(player: PlayerIn) -> Dict[str, Any]:
    players = await get_all_players()
    if any(p.name == player.name for p in players):
        raise HTTPException(status_code=400, detail='Player with this name already exists')
    new_player = GTFPlayer(player.name, stats=player.stats, player_class=player.player_class)
    players.append(new_player)
    await save_all_players(players)
    return new_player.to_dict()

@app.post('/match', summary='Update ratings after a match')
async def play_match(match: MatchIn) -> Dict[str, Any]:
    players = await get_all_players()
    team_a = [p for p in players if p.name in match.team_a]
    team_b = [p for p in players if p.name in match.team_b]
    if len(team_a) != len(match.team_a) or len(team_b) != len(match.team_b):
        raise HTTPException(status_code=400, detail='Invalid player names')
    rank_a = 1.0 - match.team_a_score
    rank_b = match.team_a_score
    system.update_ratings([team_a, team_b], [rank_a, rank_b])
    await save_all_players(players)
    return {
        'team_a': [p.to_dict() for p in team_a],
        'team_b': [p.to_dict() for p in team_b]
    }

@app.get('/antifraud', response_model=List[Dict[str, Any]], summary='Check players for smurfing/anti-fraud')
async def antifraud() -> List[Dict[str, Any]]:
    players = await get_all_players()
    suspects = system.antifraud_check(players)
    return suspects 