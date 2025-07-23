# connect_four.py (modificado)
import numpy as np
from ID3_MCTS import DecisionTree

# Depois cria e treina a árvore lá, ou importa o objeto já treinado

class ConnectFour:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.winning_length = 4  # Number of pieces in a row needed to win

    def get_board(self):
        return self.board

    def drop_piece(self, column, player):
        """Drops a piece into the specified column for the given player."""
        if self.is_valid_location(column):
            row = self.get_next_open_row(column)
            self.board[row][column] = player
            return True  # Indicate successful move
        else:
            return False  # Indicate invalid move

    def is_valid_location(self, column):
        """Checks if the specified column is a valid location to drop a piece."""
        return 0 <= column < self.cols and self.board[self.rows - 1][column] == 0

    def get_next_open_row(self, column):
        """Gets the next open row in the specified column."""
        for r in range(self.rows):
            if self.board[r][column] == 0:
                return r

    def check_win(self, player):
        """Checks if the specified player has won the game."""
        # Check horizontal
        for c in range(self.cols - self.winning_length + 1):
            for r in range(self.rows):
                if all(self.board[r][c + i] == player for i in range(self.winning_length)):
                    return True

        # Check vertical
        for c in range(self.cols):
            for r in range(self.rows - self.winning_length + 1):
                if all(self.board[r + i][c] == player for i in range(self.winning_length)):
                    return True

        # Check positively sloped diagonals
        for c in range(self.cols - self.winning_length + 1):
            for r in range(self.rows - self.winning_length + 1):
                if all(self.board[r + i][c + i] == player for i in range(self.winning_length)):
                    return True

        # Check negatively sloped diagonals
        for c in range(self.cols - self.winning_length + 1):
            for r in range(self.winning_length - 1, self.rows):
                if all(self.board[r - i][c + i] == player for i in range(self.winning_length)):
                    return True

        return False

    def is_tie(self):
        """Checks if the game is a tie."""
        return all(self.board[self.rows - 1][c] != 0 for c in range(self.cols))

    def get_valid_locations(self):
        """Returns a list of valid column indices to drop a piece."""
        return [c for c in range(self.cols) if self.is_valid_location(c)]

    def switch_player(self):
        """Switches the current player."""
        self.current_player = 3 - self.current_player  # Toggle between 1 and 2

    def get_current_player(self):
        return self.current_player

    def copy(self):
        """Creates a deep copy of the game state."""
        new_game = ConnectFour(self.rows, self.cols)
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        return new_game

    def print_board(self):
        """Prints the board to the console (for debugging)."""
        print(np.flip(self.board, 0))

    def evaluate_window(self, window, player):
        """Evaluates a window of 4 for the given player."""
        score = 0
        opponent = 3 - player

        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 4

        return score

    def score_position(self, player):
        """Scores the board for the given player."""
        score = 0

        # Score center column
        center_array = [int(i) for i in list(self.board[:, self.cols // 2])]
        center_count = center_array.count(player)
        score += center_count * 3

        # Score Horizontal
        for r in range(self.rows):
            row_array = [int(i) for i in list(self.board[r, :])]
            for c in range(self.cols - 3):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window, player)

        # Score Vertical
        for c in range(self.cols):
            col_array = [int(i) for i in list(self.board[:, c])]
            for r in range(self.rows - 3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window, player)

        # Score positive sloped diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [self.board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window, player)

        # Score negative sloped diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [self.board[r + 3 - i][c + i] for i in range(4)]
                score += self.evaluate_window(window, player)
    
        return score