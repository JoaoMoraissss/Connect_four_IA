from mcts import MCTS
from connect_four import ConnectFour
import pandas as pd
import random
from tqdm import tqdm  # Importação do tqdm

dataset = []

num_samples = 15000  # número de exemplos

for _ in tqdm(range(num_samples), desc="Gerando exemplos"):
    game = ConnectFour()
    num_moves = random.randint(0, 20)
    for _ in range(num_moves):
        valid_moves = game.get_valid_locations()
        if not valid_moves:
            break
        move = random.choice(valid_moves)
        game.drop_piece(move, game.get_current_player())
        game.switch_player()
    
    # Ignora estados terminais
    if game.check_win(1) or game.check_win(2) or game.is_tie():
        continue

    # Executa o MCTS para estados não terminais
    mcts = MCTS(game, iterations=1000)
    mcts.run()
    best_move = mcts.get_best_move()

    # Verifica se o MCTS encontrou um movimento válido
    if best_move is None:
        continue

    # Cria o exemplo para o dataset
    estado = []
    for row in game.board:
        estado.extend(row)
    exemplo = {f'cell_{i}': valor for i, valor in enumerate(estado)}
    exemplo['player'] = game.get_current_player()
    exemplo['move'] = best_move
    dataset.append(exemplo)

    # Inclui estados terminais no dataset
    if game.check_win(1) or game.check_win(2) or game.is_tie():
        estado = []
        for row in game.board:
            estado.extend(row)
        exemplo = {f'cell_{i}': valor for i, valor in enumerate(estado)}
        exemplo['player'] = game.get_current_player()
        exemplo['move'] = None  # Sem movimento válido
        dataset.append(exemplo)

# Verifica se o dataset contém exemplos antes de salvar
if len(dataset) == 0:
    print("Erro: Nenhum exemplo foi gerado. O dataset está vazio.")
else:
    df = pd.DataFrame(dataset)
    df.to_csv('connect4_mcts_dataset.csv', index=False)
    print(f"Dataset gerado com sucesso! Total de exemplos: {len(dataset)}")
