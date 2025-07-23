import pygame
import sys
import pickle
from connect_four import ConnectFour
from mcts import MCTS
import random
from ID3_MCTS import DecisionTree
from ID3_MCTS import predict_connect4_move
from ID3_MCTS import train_tree


BLUE = (120, 180, 255)       # Azul clarinho para botões
BLUE_HOVER = (170, 220, 255) # Azul mais claro para botão selecionado
BLACK = (0, 0, 0)
RED = (220, 20, 60)
YELLOW = (255, 215, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (40, 40, 40)

SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 8)
COLS = 7
ROWS = 6
WIDTH = COLS * SQUARESIZE + 220
HEIGHT = (ROWS + 1) * SQUARESIZE

pygame.init()
FONT = pygame.font.SysFont("arial", 36, bold=True)
SMALL_FONT = pygame.font.SysFont("arial", 24)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect Four IA")

# Tenta carregar o classificador ID3
try:
    with open('id3_classifier.pkl', 'rb') as f:
        id3_classifier = pickle.load(f)
except Exception:
    id3_classifier = None  # Certifique-se de importar a função predict

def draw_board(board):
    screen.fill(DARK_GRAY)
    # Tabuleiro
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, (r+1)*SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int((r+1)*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    # Peças
    for c in range(COLS):
        for r in range(ROWS):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), HEIGHT-int((r+0.5)*SQUARESIZE)), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), HEIGHT-int((r+0.5)*SQUARESIZE)), RADIUS)

def draw_menu(selected=0):
    screen.fill(DARK_GRAY)
    title = FONT.render("Connect Four IA", True, WHITE)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 60))
    options = [
        "Player vs Player",
        "Player vs IA",
        "IA vs IA",
        "ID3 vs ID3",
        "Player vs ID3",
        "ID3 vs MCTS"
    ]
    margin_y = 30
    button_height = 60
    buttons = []

    for i, opt in enumerate(options):
        text = FONT.render(opt, True, BLACK)
        btn_width = text.get_width() + 60
        btn_x = WIDTH//2 - btn_width//2
        btn_y = 180 + i * (button_height + margin_y)
        rect = pygame.Rect(btn_x, btn_y, btn_width, button_height)
        buttons.append(rect)
        color = BLUE_HOVER if i == selected else BLUE
        pygame.draw.rect(screen, color, rect, border_radius=18)
        # Texto centrado no botão
        screen.blit(text, (btn_x + (btn_width - text.get_width())//2, btn_y + (button_height - text.get_height())//2))

    pygame.display.update()
    return buttons

def draw_percentages(percentages):
    x_offset = COLS * SQUARESIZE + 10
    y_offset = 70
    title = SMALL_FONT.render("Probabilidades IA:", True, WHITE)
    screen.blit(title, (x_offset, y_offset))
    for move in sorted(percentages):
        perc = percentages[move]
        label = SMALL_FONT.render(f"Coluna {move}: {perc:.1f}%", True, YELLOW if perc > 50 else WHITE)
        screen.blit(label, (x_offset, y_offset + 30 + move*30))

def show_message(msg, color=WHITE):
    pygame.draw.rect(screen, DARK_GRAY, (0, 0, WIDTH, SQUARESIZE))
    text = FONT.render(msg, True, color)
    screen.blit(text, (WIDTH//2 - text.get_width()//2, 10))
    pygame.display.update()

def menu_loop():
    selected = 0
    buttons = draw_menu(selected)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(buttons)
                    buttons = draw_menu(selected)
                if event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(buttons)
                    buttons = draw_menu(selected)
                if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    return selected
            if event.type == pygame.MOUSEMOTION:
                mx, my = pygame.mouse.get_pos()
                for i, rect in enumerate(buttons):
                    if rect.collidepoint(mx, my):
                        if selected != i:
                            selected = i
                            buttons = draw_menu(selected)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for i, rect in enumerate(buttons):
                    if rect.collidepoint(mx, my):
                        return i
def get_valid_moves(game):
    return [col for col in range(COLS) if game.is_valid_location(col)]

tree = train_tree()

def game_loop(mode):
    game = ConnectFour(ROWS, COLS)
    turn = 0  # 0: Player 1, 1: Player 2 / IA / ID3 / MCTS
    game_over = False
    mcts = None
    percentages = {}

    draw_board(game.get_board())
    show_message("Vermelho começa!" if mode != 2 else "IA 1 a jogar...")
    pygame.display.update()

    while not game_over:
        current_player = 1 if turn == 0 else 2
        board = game.get_board()

        # Modos que envolvem ID3 - presumo que tens uma função 'predict_connect4_move' implementada
        if mode == 3:  # ID3 vs ID3
            pygame.time.wait(400)
            valid_moves = get_valid_moves(game)
            col = predict_connect4_move(tree, board, current_player, valid_moves)
            if col is not None and game.is_valid_location(col):
                game.drop_piece(col, current_player)
                draw_board(game.get_board())
                show_message(f"ID3 {current_player} jogou na coluna {col}")
                pygame.display.update()

                if game.check_win(current_player):
                    show_message(f"ID3 {current_player} venceu!", YELLOW if current_player == 2 else RED)
                    print(f"ID3 {current_player} venceu!")
                    game_over = True
                elif game.is_tie():
                    show_message("Empate!", WHITE)
                    print("Empate!")
                    game_over = True
                else:
                    game.switch_player()
                    turn = 1 - turn
            continue

        elif mode == 4:  # Player vs ID3
            if turn == 0:  # Vez do ID3
                pygame.time.wait(400)
                valid_moves = get_valid_moves(game)
                col = predict_connect4_move(tree, board, 1, valid_moves)
                if col is not None and game.is_valid_location(col):
                    game.drop_piece(col, 1)
                    draw_board(game.get_board())
                    show_message(f"ID3 jogou na coluna {col}")
                    pygame.display.update()

                    if game.check_win(1):
                        show_message("ID3 venceu!", RED)
                        print("ID3 venceu!")
                        game_over = True
                    elif game.is_tie():
                        show_message("Empate!", WHITE)
                        print
                        game_over = True
                    else:
                        game.switch_player()
                        show_message("Vez do jogador!", WHITE)
                        turn = 1
            else:  # Vez do jogador
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        col = event.pos[0] // SQUARESIZE
                        if 0 <= col < COLS and game.is_valid_location(col):
                            game.drop_piece(col, 2)
                            draw_board(game.get_board())

                            if game.check_win(2):
                                show_message("Jogador venceu!", YELLOW)
                                print("Jogador venceu!")
                                game_over = True
                            elif game.is_tie():
                                show_message("Empate!", WHITE)
                                print("Empate!")
                                game_over = True
                            else:
                                game.switch_player()
                                turn = 0
                            pygame.display.update()
            if game_over:
                print("Fim do jogo!")
                pygame.time.wait(2200)
                return
            continue

        elif mode == 5:  # ID3 vs MCTS
            if turn == 0:  # ID3 joga
                pygame.time.wait(400)
                valid_moves = get_valid_moves(game)
                col = predict_connect4_move(tree, board, 1, valid_moves)
                if col is not None and game.is_valid_location(col):
                    game.drop_piece(col, 1)
                    draw_board(game.get_board())
                    show_message(f"ID3 jogou na coluna {col}")
                    pygame.display.update()

                    if game.check_win(1):
                        show_message("ID3 venceu!", RED)
                        print("ID3 VENCEU")
                        game_over = True
                    elif game.is_tie():
                        show_message("Empate!", WHITE)
                        game_over = True
                    else:
                        game.switch_player()
                        turn = 1
            else:  # MCTS joga
                pygame.time.wait(400)
                valid_moves = get_valid_moves(game)
                if valid_moves:
                    mcts = MCTS(game, iterations=400)
                    mcts.run()
                    col = mcts.get_best_move()
                    if col is not None and game.is_valid_location(col):
                        game.drop_piece(col, 2)
                        draw_board(game.get_board())
                        show_message(f"MCTS jogou na coluna {col}")
                        pygame.display.update()

                        if game.check_win(2):
                            show_message("MCTS venceu!", YELLOW)
                            print("MCTS VENCEU")
                            game_over = True
                        elif game.is_tie():
                            show_message("Empate!", WHITE)
                            print("Empate!")
                            game_over = True
                        else:
                            game.switch_player()
                            turn = 0
            if game_over:
                pygame.time.wait(2200)
                return
            continue

        elif mode == 2:  # IA vs IA
            pygame.time.wait(400)
            mcts = MCTS(game, iterations=400)
            mcts.run()
            col = mcts.get_best_move()
            if col is not None and game.is_valid_location(col):
                game.drop_piece(col, current_player)
                draw_board(game.get_board())
                show_message(f"MCTS {current_player} jogou na coluna {col}")
                pygame.display.update()

                if game.check_win(current_player):
                    show_message(f"MCTS{current_player} venceu!", YELLOW if current_player == 2 else RED)
                    print(f"MCTS {current_player} venceu!")
                    game_over = True
                elif game.is_tie():
                    show_message("Empate!", WHITE)
                    print("Empate!")
                    game_over = True
                else:
                    game.switch_player()
                    turn = 1 - turn
            continue

        # Modos tradicionais: PvP e PvIA
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if mode == 0:  # Player vs Player
                if event.type == pygame.MOUSEMOTION:
                    draw_board(game.get_board())
                    x = event.pos[0]
                    pygame.draw.circle(screen, RED if turn == 0 else YELLOW, (x, SQUARESIZE // 2), RADIUS)
                    pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    col = event.pos[0] // SQUARESIZE
                    player = 1 if turn == 0 else 2
                    if 0 <= col < COLS and game.is_valid_location(col):
                        game.drop_piece(col, player)
                        draw_board(game.get_board())
                        if game.check_win(player):
                            show_message(f"Jogador {player} venceu!", RED if player == 1 else YELLOW)
                            print(f"Jogador {player} venceu!")
                            game_over = True
                        elif game.is_tie():
                            show_message("Empate!", WHITE)
                            print("Empate!")
                            game_over = True
                        else:
                            game.switch_player()
                            turn = 1 - turn
                        pygame.display.update()

            elif mode == 1:  # Player vs IA
                if turn == 0:  # Jogador
                    if event.type == pygame.MOUSEMOTION:
                        draw_board(game.get_board())
                        x = event.pos[0]
                        pygame.draw.circle(screen, RED, (x, SQUARESIZE // 2), RADIUS)
                        pygame.display.update()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        col = event.pos[0] // SQUARESIZE
                        if 0 <= col < COLS and game.is_valid_location(col):
                            game.drop_piece(col, 1)
                            draw_board(game.get_board())
                            if game.check_win(1):
                                show_message("Jogador venceu!", RED)
                                print("Jogador venceu")
                                game_over = True
                            elif game.is_tie():
                                show_message("Empate!", WHITE)
                                print("Empate!")
                                game_over = True
                            else:
                                game.switch_player()
                                turn = 1
                            pygame.display.update()
                else:  # IA
                    show_message("MCTS a pensar...", YELLOW)
                    pygame.display.update()
                    mcts = MCTS(game, iterations=700)
                    mcts.run()
                    percentages = mcts.get_win_percentages()
                    col = mcts.get_best_move()
                    pygame.time.wait(400)
                    if col is not None and game.is_valid_location(col):
                        game.drop_piece(col, 2)
                        draw_board(game.get_board())
                        draw_percentages(percentages)
                        show_message(f"MCTS jogou na coluna {col}")
                        pygame.display.update()

                        if game.check_win(2):
                            show_message("MCTS venceu!", YELLOW)
                            print(f"MCTS VENCEU")
                            game_over = True
                        elif game.is_tie():
                            show_message("Empate!", WHITE)
                            print("Empate!")
                            game_over = True
                        else:
                            game.switch_player()
                            turn = 0
            if game_over:
                pygame.time.wait(2200)
                return


if __name__ == "__main__":
    while True:
        mode = menu_loop()  # 0: PvP, 1: PvIA, 2: IAvsIA, 3: ID3vsID3, 4: ID3vsEU, 5: ID3vsID3+MCTS
        game_loop(mode)