from mcts import MCTS
from connect_four import ConnectFour
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset = []
num_samples = 500  # número de exemplos
total_moves = 0      # soma total das jogadas
num_games = 0        # número de partidas simuladas
moves_per_game = []  # lista para guardar o número de jogadas por partida

for _ in tqdm(range(num_samples), desc="Gerando exemplos"):
    game = ConnectFour()
    num_moves = random.randint(0, 20)
    moves_this_game = 0  # contador de jogadas desta partida

    for _ in range(num_moves):
        valid_moves = game.get_valid_locations()
        if not valid_moves:
            break
        move = random.choice(valid_moves)
        game.drop_piece(move, game.get_current_player())
        game.switch_player()
        moves_this_game += 1

    # Ignora estados terminais
    if game.check_win(1) or game.check_win(2) or game.is_tie():
        continue

    # Executa o MCTS para estados não terminais
    mcts = MCTS(game, iterations=1000)
    mcts.run()
    best_move = mcts.get_best_move()

    if best_move is None:
        continue

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
        exemplo['move'] = None
        dataset.append(exemplo)

    # Atualiza estatísticas
    total_moves += moves_this_game
    num_games += 1
    moves_per_game.append(moves_this_game)

if len(dataset) == 0:
    print("Erro: Nenhum exemplo foi gerado. O dataset está vazio.")
else:
    df = pd.DataFrame(dataset)
    df.to_csv('connect4_mcts_dataset.csv', index=False)
    print(f"Dataset gerado com sucesso! Total de exemplos: {len(dataset)}")
    if num_games > 0:
        media_jogadas = total_moves / num_games
        print(f"Número médio de jogadas por partida: {media_jogadas:.2f}")
    else:
        print("Nenhuma partida válida foi simulada.")

    # Gera o gráfico
    plt.figure(figsize=(8,5))
    plt.hist(moves_per_game, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribuição do Número de Jogadas por Partida')
    plt.xlabel('Número de Jogadas')
    plt.ylabel('Frequência')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(media_jogadas, color='red', linestyle='dashed', linewidth=2, label=f'Média = {media_jogadas:.2f}')
    plt.legend()
    plt.savefig('distribuicao_jogadas_por_partida.png')
    plt.show()
