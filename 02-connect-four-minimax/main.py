import copy
import math
import random

EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'
ROWS = 6
COLS = 7

class ConnectFour:
    """
    Class for game Connect 4

    """
    def __init__(self, ai_piece, user_piece):
        self.board = [[EMPTY] * COLS for _ in range(ROWS)]
        self.current_player = PLAYER_X
        self.ai_piece = ai_piece
        self.user_piece = user_piece

    def print_board(self):
        for row in self.board:
            print('|'.join(row))
        print('-' * (COLS * 2 - 1))
        print(' '.join(str(i) for i in range(COLS)))

    # Implement any additional functions needed here

    def evaluate_window(self, window, piece):
        """
        Evaluation of given window. Helper function to evaluate the separate parts of the board called windows

        Parameters:
        - window: list containing values of evaluated window
        - piece: PLAYER_X or PLAYER_O depending on which player's position we evaluate

        Returns:
        - score of the window
        """
        score = 0
        not_piece = PLAYER_O if piece == PLAYER_X else PLAYER_X

        my_pieces = window.count(piece)
        empty_spots = window.count(EMPTY)
        opp_pieces = window.count(not_piece)

        if my_pieces == 4:
            score += 1000
        elif my_pieces == 3 and empty_spots == 1: 
            score += 10
        elif my_pieces == 2 and empty_spots == 2:
            score += 2

        if opp_pieces == 4: 
            score -= 1000
        if opp_pieces == 3 and empty_spots == 1:
            score -= 80
        elif opp_pieces == 2 and empty_spots == 2:
            score -= 10

        return score 

    def evaluate_position(self, board, piece):
        """
        Evaluation of position
        Parameters:
        - board: 2d matrix representing evaluated state of the board
        - piece: PLAYER_X or PLAYER_O depending on which player's position we evaluate

        Returns:
        - score of the position

        """
        score = 0

        center_col = COLS // 2

        #This is supposed to make the program prioritize the center 
        center_array = [board[i][COLS//2] for i in range(ROWS)]
        center_count = center_array.count(piece)
        score += center_count * 3

        if COLS >= 3:
            left_center_array = [board[i][center_col - 1] for i in range(ROWS)]
            right_center_array = [board[i][center_col + 1] for i in range(ROWS)]
            score += left_center_array.count(piece) * 2
            score += right_center_array.count(piece) * 2

        #Horizontal scan
        for c in range (COLS-3):
            for r in range (ROWS): 
                window = [board[r][c], board[r][c+1], board[r][c+2], board[r][c+3]]
                score += self.evaluate_window(window, piece)
        #Vertical scan
        for c in range(COLS):
            for r in range(ROWS - 3):
                window = [board[r][c], board[r+1][c], board[r+2][c], board[r+3][c]]
                score += self.evaluate_window(window, piece)
        #/
        for c in range(COLS - 3):
            for r in range(ROWS - 3):
                window = [board[r][c], board[r+1][c+1], board[r+2][c+2], board[r+3][c+3]]
                score += self.evaluate_window(window, piece)

        #\
        for c in range(COLS - 3):
            for r in range(3, ROWS):
                window = [board[r][c], board[r-1][c+1], board[r-2][c+2], board[r-3][c+3]]
                score += self.evaluate_window(window, piece)
        return score
    
    def get_valid_columns(self, board): 
        """ 
        Returns the indicies of columns where it is possible to place the piece
        """
        idx = [] 
        for i in range(COLS):
            if board [0][i] == EMPTY:
                idx.append(i)
        return idx

    def winning_move(self, board, piece): 
        """
        Returns boolean value indicating if the given peace Won or Not
        """
        #Horizontal scan 
        for c in range(COLS - 3):
            for r in range(ROWS):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True
                
        #Vertical scan 
        for c in range(COLS):
            for r in range(ROWS - 3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True
                
        #/ 
        for c in range(COLS - 3):
            for r in range(ROWS - 3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True
        #\        
        for c in range(COLS - 3):
            for r in range(3, ROWS):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True        
        return False 
    

    def is_terminal_node(self, board): 
        """
        Returns boolean value checking if the game is finished, ie. if someone won or there is no more places for pieces
        """
        return self.winning_move(board, self.user_piece) or self.winning_move(board, self.ai_piece) or len(self.get_valid_columns(board)) == 0

    def drop_piece(self, b_copy, col, piece):
        """
        Places a piece in the desired column
        """
        for i in range(ROWS-1, -1, -1):
            if b_copy[i][col] == EMPTY: 
                b_copy[i][col] = piece 
                break
        return None

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        """
        Minimax with alpha-beta pruning algorithm

        Parameters:
        - board: 2d matrix representing the state, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2) 
        - depth: depth
        - maximizing_player: boolean which is equal to True when the player tries to maximize the score
        - alpha: alpha variable for pruning
        - beta: beta variable for pruning

        Returns:
        - Best value 
        - Best move found

        """
        valid_columns = self.get_valid_columns(board)
        is_terminal = self.is_terminal_node(board)

        if is_terminal: 
            if self.winning_move(board, self.ai_piece): 
                return (None, 1000000)
            elif self.winning_move(board, self.user_piece): 
                return (None, -1000000)
            else: 
                return (None, 0) #Draw 
        elif depth == 0: 
            return None, self.evaluate_position(board, self.ai_piece)
                
        #AI will always be the maximizing player 
        if maximizing_player: 
            value = -math.inf
            best_col = random.choice(valid_columns)

            for col in valid_columns:
                #copy of the board using list comprehension
                b_copy = [row[:] for row in board] 
                #make a move
                self.drop_piece(b_copy, col, self.ai_piece) 
                new_value = self.minimax(b_copy, depth-1, False, alpha, beta)[1]

                if new_value > value: 
                    value = new_value 
                    best_col = col
                
                #prune the remaining branches 
                alpha = max(alpha, value) 
                if alpha >= beta: 
                    break

            return best_col, value

        else: 
            value = math.inf 
            best_col = random.choice(valid_columns) 

            for col in valid_columns: 
                b_copy = [row[:] for row in board] 

                self.drop_piece(b_copy, col, self.user_piece) 
                new_value = self.minimax(b_copy, depth-1, True, alpha, beta)[1]

                if new_value < value: 
                    value = new_value 
                    best_col = col 

                #prune the remaining branches
                beta = min(beta, value)
                if alpha >= beta: 
                    break

            return best_col, value 

    def run_tests(self):
        """Test cases for the AI"""
        
        # Test 1: AI should play the winning move (AI has 3 in a row horizontally)
        print("\n=== Test 1: AI plays winning move ===")
        test_board = [[EMPTY] * COLS for _ in range(ROWS)]
        test_board[5][0] = self.ai_piece
        test_board[5][1] = self.ai_piece
        test_board[5][2] = self.ai_piece
        # Expected: AI plays column 3 to win
        best_col, _ = self.minimax(test_board, 10, True, -math.inf, math.inf)
        print(f"Expected col 3, got col {best_col} -> {'PASS' if best_col == 3 else 'FAIL'}")

        # Test 2: AI should block the user from winning
        print("\n=== Test 2: AI blocks user ===")
        test_board = [[EMPTY] * COLS for _ in range(ROWS)]
        test_board[5][0] = self.user_piece
        test_board[5][1] = self.user_piece
        test_board[5][2] = self.user_piece
        # Expected: AI plays column 3 to block
        best_col, _ = self.minimax(test_board, 5, True, -math.inf, math.inf)
        print(f"Expected col 3, got col {best_col} -> {'PASS' if best_col == 3 else 'FAIL'}")

        # Test 3: AI should prefer center column on empty board
        print("\n=== Test 3: AI prefers center on empty board ===")
        test_board = [[EMPTY] * COLS for _ in range(ROWS)]
        best_col, _ = self.minimax(test_board, 5, True, -math.inf, math.inf)
        print(f"Expected col 3, got col {best_col} -> {'PASS' if best_col == 3 else 'FAIL'}")

        # Test 4: AI blocks vertical win
        print("\n=== Test 4: AI blocks vertical win ===")
        test_board = [[EMPTY] * COLS for _ in range(ROWS)]
        test_board[5][2] = self.user_piece
        test_board[4][2] = self.user_piece
        test_board[3][2] = self.user_piece
        # Expected: AI plays column 2 to block
        best_col, _ = self.minimax(test_board, 5, True, -math.inf, math.inf)
        print(f"Expected col 2, got col {best_col} -> {'PASS' if best_col == 2 else 'FAIL'}")

        # Test 5: AI prefers winning over blocking
        print("\n=== Test 5: AI prefers winning over blocking ===")
        test_board = [[EMPTY] * COLS for _ in range(ROWS)]
        # User is about to win horizontally
        test_board[5][0] = self.user_piece
        test_board[5][1] = self.user_piece
        test_board[5][2] = self.user_piece
        # AI is also about to win vertically
        test_board[5][6] = self.ai_piece
        test_board[4][6] = self.ai_piece
        test_board[3][6] = self.ai_piece
        # Expected: AI plays column 6 to win rather than 3 to block
        best_col, _ = self.minimax(test_board, 5, True, -math.inf, math.inf)
        print(f"Expected col 6, got col {best_col} -> {'PASS' if best_col == 6 else 'FAIL'}")

        # Test 6: AI blocks a horizontal gap
        print("\n=== Test 6: AI blocks horizontal gap ===")
        test_board = [[EMPTY] * COLS for _ in range(ROWS)]
        test_board[5][1] = self.user_piece
        test_board[5][2] = self.user_piece
        test_board[5][4] = self.user_piece
        # Expected: AI plays column 3 to block the trap
        best_col, _ = self.minimax(test_board, 5, True, -math.inf, math.inf)
        print(f"Expected col 3, got col {best_col} -> {'PASS' if best_col == 3 else 'FAIL'}")


def main():
    """
    Main game loop implementation. Player1 should play first with 'X', player2 plays second with 'O'
    """

    print("Welcome to the game of connect four")  

    while True:
        choice = input("Do you want to be the first or second player? (type 1 or 2): ")
    
        if choice == "1": 
            user_piece = PLAYER_X 
            ai_piece = PLAYER_O
            break 
            
        elif choice == "2": 
            user_piece = PLAYER_O
            ai_piece = PLAYER_X
            break
            
        else: 
            print("Invalid input. Please type  1 or 2.")

    game = ConnectFour(ai_piece, user_piece)

    
    
     = -math.inf
    beta = math.inf

    # Run tests before starting
    while True:
        run_tests_choice = input("Run test cases before playing? (y/n): ")
        if run_tests_choice == "y":
            game.run_tests()
            break
        elif run_tests_choice == "n":
            break
        else:
            print("Invalid input. Please type y or n.")

    print("\nGame starts! X plays first.")
    game.print_board()

    while True:
        # --- USER TURN ---
        if game.current_player == user_piece:
            while True:
                try:
                    col = int(input(f"Your turn ({user_piece}). Choose a column (0-6): "))
                    if col not in range(COLS):
                        print("Column must be between 0 and 6.")
                    elif col not in game.get_valid_columns(game.board):
                        print("Column is full. Choose another one.")
                    else:
                        break
                except ValueError:
                    print("Invalid input. Please enter a number.")

            game.drop_piece(game.board, col, user_piece)
            game.print_board()

            if game.winning_move(game.board, user_piece):
                print("You win! Congratulations!")
                break

        # --- AI TURN ---
        else:
            print("AI is thinking...")
            best_col, _ = game.minimax(game.board, 6, True, -math.inf, math.inf)
            game.drop_piece(game.board, best_col, ai_piece)
            print(f"AI played column {best_col}")
            game.print_board()

            if game.winning_move(game.board, ai_piece):
                print("AI wins! Better luck next time.")
                break

        # --- CHECK DRAW ---
        if len(game.get_valid_columns(game.board)) == 0:
            print("It's a draw!")
            break

        # --- SWITCH PLAYER ---
        game.current_player = PLAYER_O if game.current_player == PLAYER_X else PLAYER_X

 
if __name__ == "__main__":
    main()
