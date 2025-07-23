import pandas as pd
import time
from tqdm import tqdm

from ID3_MCTS import predict_connect4_move
from connect_four import ConnectFour

from mcts import MCTS
import pickle
from ID3_MCTS import DecisionTree, train_tree, predict_connect4_move
import pickle

num_games = 50 # Número de partidas a simular
csv_filename = "tempos_mcts_vs_id3.csv"
resultados = []
import pickle
with open("connect4_tree.pkl", "rb") as f:
    tree = pickle.load(f)


for _ in tqdm(range(num_games), desc=" partidas"):
    game = ConnectFour()
    moves_this_game = 0
    tempos_mcts = []
    tempos_id3 = []
    vencedor = 0

    while not game.check_win(1) and not game.check_win(2) and not game.is_tie():
        if game.get_current_player() == 1:
            # MCTS joga
            start = time.time()
            mcts = MCTS(game, iterations=400)
            mcts.run()
            move = mcts.get_best_move()
            end = time.time()
            tempos_mcts.append(end - start)
        else:
            # ID3 joga
            start = time.time()
            valid_moves = game.get_valid_locations()
            move = predict_connect4_move(tree, game.get_board(), game.get_current_player(), valid_moves)
            end = time.time()
            tempos_id3.append(end - start)
        game.drop_piece(move, game.get_current_player())
        game.switch_player()
        moves_this_game += 1

    # Determina vencedor
    if game.check_win(1):
        vencedor = 1
    elif game.check_win(2):
        vencedor = 2
    else:
        vencedor = 0  # Empate

    tempo_medio_mcts = sum(tempos_mcts) / len(tempos_mcts) if tempos_mcts else 0
    tempo_medio_id3 = sum(tempos_id3) / len(tempos_id3) if tempos_id3 else 0

    resultados.append({
        "num_jogadas": moves_this_game,
        "vencedor": vencedor,
        "tempo_medio_mcts": tempo_medio_mcts,
        "tempo_medio_id3": tempo_medio_id3
    })

# Salva CSV
df = pd.DataFrame(resultados)
df.to_csv(csv_filename, index=False)
print(f"CSV '{csv_filename}' gerado com sucesso!")

# Estatísticas rápidas e gráfico dos tempos
media_mcts = df['tempo_medio_mcts'].mean()
media_id3 = df['tempo_medio_id3'].mean()
print(f"Tempo médio MCTS: {media_mcts:.4f} s")
print(f"Tempo médio ID3: {media_id3:.4f} s")

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.bar(['MCTS', 'ID3'], [media_mcts, media_id3 * 1000], color=['royalblue', 'orange'])
plt.ylabel('Tempo médio por jogada (ms)')
plt.title('Tempo Médio de Decisão por Jogada (MCTS vs ID3)')
plt.savefig('tempo_medio_mcts_vs_id3_ms.png')
plt.show()
