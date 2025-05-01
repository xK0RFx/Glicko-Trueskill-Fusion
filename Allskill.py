import argparse
import logging
import json

from src.system import GTFSystem
from src.models import GTFPlayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    parser = argparse.ArgumentParser(description='GTF System CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # load command
    p_load = subparsers.add_parser('load', help='Load players from JSON')
    p_load.add_argument('players_file', help='Path to players JSON file')

    # save command
    p_save = subparsers.add_parser('save', help='Save players to JSON')
    p_save.add_argument('players_file', help='Path to players JSON file')

    # antifraud command
    p_anti = subparsers.add_parser('antifraud', help='Detect smurfing players')
    p_anti.add_argument('players_file', help='Path to players JSON file')
    p_anti.add_argument('--min-matches', type=int, default=10, help='Minimum matches threshold')
    p_anti.add_argument('--growth', type=float, default=400, help='Rating growth threshold')
    p_anti.add_argument('--rd-threshold', type=float, default=80, help='RD threshold')

    # calibrate command
    p_calib = subparsers.add_parser('calibrate', help='Calibrate system parameters')
    p_calib.add_argument('history_file', help='Path to match history JSON file')

    # update command
    p_update = subparsers.add_parser('update', help='Update ratings for a two-team match')
    p_update.add_argument('team_a_file', help='JSON file with team A players')
    p_update.add_argument('team_b_file', help='JSON file with team B players')
    p_update.add_argument('score', type=float, choices=[0.0, 0.5, 1.0], help='Score for team A (0, 0.5, 1)')
    p_update.add_argument('output_file', help='Path to save updated players JSON')

    args = parser.parse_args()
    system = GTFSystem()

    if args.command == 'load':
        players = system.load_players(args.players_file)
        logging.info(f'Loaded {len(players)} players from {args.players_file}')
    elif args.command == 'save':
        players = system.load_players(args.players_file)
        system.save_players(players, args.players_file)
        logging.info(f'Saved {len(players)} players to {args.players_file}')
    elif args.command == 'antifraud':
        players = system.load_players(args.players_file)
        suspects = system.antifraud_check(players)
        print(json.dumps(suspects, ensure_ascii=False, indent=2))
    elif args.command == 'calibrate':
        with open(args.history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        best_params, best_rmse = system.calibrate(history)
        print(json.dumps({'best_params': best_params, 'best_rmse': best_rmse}, ensure_ascii=False, indent=2))
    elif args.command == 'update':
        # Load teams from files using system
        team_a = system.load_players(args.team_a_file)
        team_b = system.load_players(args.team_b_file)
        # Convert score to ranks: rank = 1 - outcome
        rank_a = 1.0 - args.score
        rank_b = args.score
        # Update ratings
        system.update_ratings([team_a, team_b], [rank_a, rank_b])
        # Combine updated players and save
        updated = [p.to_dict() for p in team_a + team_b]
        system.save_players(updated, args.output_file)
        logging.info(f'Updated ratings saved to {args.output_file}')
    else:
        parser.print_help()

if __name__ == '__main__':
    main()