import math
import random
import numpy as np

class Node:
    def __init__(self, game, parent=None, move=None):
        self.game = game  # Estado do jogo
        self.parent = parent
        self.move = move  # Jogada que levou a este estado
        self.children = {}  # move: Node
        self.wins = 0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.game.get_valid_locations())

    def is_terminal(self):
        return self.game.check_win(1) or self.game.check_win(2) or self.game.is_tie()

    def uct_value(self, exploration_constant=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')  # Priorizar nós não visitados
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        # Seleciona o filho com maior UCT
        return max(self.children.values(), key=lambda c: c.uct_value())

    def expand(self):
        # Adiciona todos os filhos válidos que ainda não existem
        valid_moves = self.game.get_valid_locations()
        for move in valid_moves:
            if move not in self.children:
                new_game = self.game.copy()
                new_game.drop_piece(move, new_game.get_current_player())
                new_game.switch_player()
                self.children[move] = Node(new_game, parent=self, move=move)

    def rollout(self):
        # Simulação aleatória até ao fim do jogo
        current_game = self.game.copy()
        current_player = current_game.get_current_player()

        while not current_game.check_win(1) and not current_game.check_win(2) and not current_game.is_tie():
            valid_moves = current_game.get_valid_locations()
            move = random.choice(valid_moves)
            current_game.drop_piece(move, current_player)
            current_game.switch_player()
            current_player = current_game.get_current_player()

        if current_game.check_win(2):
            return 2
        elif current_game.check_win(1):
            return 1
        else:
            return 0  # empate

    def backpropagate(self, result):
        self.visits += 1
        # Se o resultado é vitória do jogador que fez a jogada para chegar neste nó, soma 1
        # Caso contrário, soma 0
        if self.parent is None:
            # Raiz: não tem jogador associado, ignora
            pass
        else:
            player_who_moved = 3 - self.game.get_current_player()  # jogador que fez a jogada para chegar aqui
            if result == player_who_moved:
                self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, game, iterations=100000):
        self.root = Node(game)
        self.iterations = iterations

    def select(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.select_child()
        return node

    def expand(self, node):
        node.expand()
        # Retorna um filho aleatório ainda não visitado
        unvisited = [child for child in node.children.values() if child.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        else:
            # Se todos visitados, retorna qualquer filho
            return random.choice(list(node.children.values()))

    def run(self):
        for _ in range(self.iterations):
            leaf = self.select(self.root)
            result = leaf.rollout()
            leaf.backpropagate(result)

    def get_best_move(self):
        # Escolhe o filho com mais visitas
        best_move = None
        best_visits = -1
        for move, child in self.root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
        return best_move

    def make_move(self, move):
        # Avança a raiz para o estado da jogada feita
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            new_game = self.root.game.copy()
            new_game.drop_piece(move, new_game.get_current_player())
            new_game.switch_player()
            self.root = Node(new_game)

    def get_win_percentages(self):
        percentages = {}
        for move, child in self.root.children.items():
            if child.visits > 0:
                percentages[move] = (child.wins / child.visits) * 100
            else:
                percentages[move] = 0.0
        return percentages

