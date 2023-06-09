import random

class Game:

    def __init__(self, player1, player2):
        """Initialize a new game."""
        self.player1 = player1
        self.player2 = player2
        self.turn = player1
        self.winner = None
        self.turn_number = 1

        # initialize the board
        self.board = Board(self)
        self.moves = []

        # set active games
        player1.play_game(self)
        player2.play_game(self)

    def make_move(self, move):
        """Make a move on the board.

        Passes the move to the board, and then updates whos turn it is
        
        Parameters:
        ------------
        move : Move
            The move to be made.
        """
        if self.winner is not None:
            print(self.winner)
            raise ValueError("Game is over")
        
        # for the first for turns check:
        # 1. if the bee has not been placed the move is a placement move
        # 2. if the bee has not been placed within the first 3 moves decline any non-placement moves for the 4th turn
        if self.turn_number < 5:
            bee = self.board.get_bee(move.player)
            if move.piece.position is not None:
                # if bee.position is None:
                if not bee.placed:
                    print("You cannot move before placing the bee")
                    return False
            if self.turn_number == 4:
                if not bee.placed and move.piece.__str__() != "Bee":
                    print("You have to place your bee within the first 4 turns")
                    return False

        res = move.piece.try_move(move)
        if res:
            move.piece.make_move(move)
            self.moves.append(move)
            self.board.generate_board_state()
            # check if game is over
            result = self.board.check_game_over()
            if result is not False:
                self.winner = result
                print("Game Over")
            self.turn = self.player2 if self.turn == self.player1 else self.player1
            print(self.turn)
            if self.turn == self.player1:
                self.turn_number += 1
        
        # print(self.all_legal_moves())
        return res
    
    def all_legal_moves(self):
        """Return all legal moves for a given player."""
        player = self.turn
        moves = []
        edges = self.board.edges()
        movement = "Allowed"

        if self.turn_number < 5:
            bee = self.board.get_bee(self.turn)
            if bee.position is None:
                movement = "Illegal"
            if bee.position is None and self.turn_number == 4:
                movement = "Place bee"
                print("You have to place your bee within the first 4 turns")
            

        for piece in self.board.pieces:
            if piece.player == player:
                moves += piece.legal_moves(edges, movement)
        return moves

class Board:
    """Represents the board of a hive game.
    
    Attributes:
    ------------
    game : Game
        Referece to the corresponding game object. 
    """

    # here you can play with the number and types of pieces
    game_mode = [('Bee', 1), ('Ant',3), ('Beetle', 2), ('Spider', 2), ('Grasshopper', 3)]
    # game_mode = [('Bee', 1), ('Beetle', 2)]
    
    def __init__(self, game: Game):
        """Initialize a new board.
        
        Given the game mode it generates the pieces for each player.
        """
        self.pieces = []
        self.board_state = []
        self.game = game

        # get pieces for each player
        for player in [self.game.player1, self.game.player2]:
            for p in self.game_mode:
                if p[0] == 'Bee':
                    for i in range(p[1]):
                        self.pieces.append(Bee(player, self.game))
                elif p[0] == 'Ant':
                    for i in range(p[1]):
                        self.pieces.append(Ant(player, self.game))
                elif p[0] == 'Beetle':
                    for i in range(p[1]):
                        self.pieces.append(Beetle(player, self.game))
                elif p[0] == 'Spider':
                    for i in range(p[1]):
                        self.pieces.append(Spider(player, self.game))
                elif p[0] == 'Grasshopper':
                    for i in range(p[1]):
                        self.pieces.append(Grasshopper(player, self.game))

    def get_bee(self, player):
        """Returns the bee of the given player."""
        for piece in self.pieces:
            if piece.__str__() == 'Bee' and piece.player == player:
                return piece
        return None

    def get_piece(self, position, player=None, piece_type = None):
        """Return the piece at the given position.
        
        Optionally, the player and piece type can be specified."""
        for piece in self.pieces:
            if piece.position == position and piece.mounted == False:
                if player == None or (player == piece.player and piece_type == piece.__str__()):
                    return piece
        if position == None:
            raise RuntimeError("No more pieces of this type available")
        return None

    @staticmethod
    def neighboring_hexes(position):
        """Return the neighboring hexes of a given hex.
        
        Position needs to be in hex coordinates."""
        return [[position[i] + adj[i] for i in range(len(position))] for adj in Piece.adjacent] 

    def get_edge(self, position):
        """Returns the edge of a given hex. (Position in hex coordinates)"""
        neighbors = Board.neighboring_hexes(position)
        edge = [None for i in range(len(neighbors))]
        for i, neighbor in enumerate(neighbors):
            piece = self.get_piece(neighbor)
            if piece != None:
                edge[i] = piece
        return edge

    def check_game_over(self):
        # the game is over if either bee is fully surrounded
        # the game end in a draw if both bees are fully surrounded
        for piece in self.pieces:
            if piece.__str__() == 'Bee':
                if piece.player == self.game.player1:
                    bee1 = piece
                else:
                    bee2 = piece

        if bee1.num_neighbors() == 6:
            if bee2.num_neighbors() == 6:
                return 'Draw'
            return self.game.player2
        elif bee2.num_neighbors() == 6:
            return self.game.player1
        return False

    def is_connected(self, move=None):
        """Check if the board is connected."""
        pieces = []
        for piece in self.pieces:
            if move!=None and piece == move.piece:
                move.piece.visited = True
                continue
            piece.visited = False
            if piece.position is not None:
                pieces.append(piece)
        
        # if pieces is empty raise: This is the first piece
        if len(pieces) == 0:
            print("piece is the first piece")
            return True

        # basically a depth first search
        # otherwise, check if the hive is connected
        stack = [pieces[0]]
        connected_count = 0
        while len(stack) > 0:
            piece = stack.pop()
            connected_count += 1
            piece.visited = True
            neighbors = [piece for piece in piece.edge if piece != None]
            for adj in neighbors:
                if adj.visited == False and adj not in stack:
                    stack.append(adj)
        
        if connected_count == len(pieces):
            return True

        return False

    def generate_board_state(self):
        """Generates a board state from the current board from the piece positions."""
        self.board_state = [[piece.__repr__(), piece.position, piece.edge] for piece in self.pieces if piece.position is not None]
        # for line in self.board_state:
        #     print(line)

    def make_move(self, move):
        """Passes the legal move to the piece and updates the board state accordingly."""
        if self.legal_board(move):
            res = move.piece.move(move.position, move.edge)
            if res:
                self.generate_board_state()
                for line in self.board_state:
                    print(line)
            return res
        return False

    def edges(self):
        """Returns all edges of the board."""
        edges = []
        for piece in self.pieces:
            if piece.position is not None:
                reach = Board.neighboring_hexes(piece.position)
                for i in range(len(reach)):
                    if piece.edge[i] == None:
                        edges.append(reach[i])
        edges = list(set([tuple(edge) for edge in edges]))
        edges = [list(edge) for edge in edges]
        if len(edges) == 0:
            edges.append([0,0,0])
        return edges
    
    def check_contested(self, edge, player):
        """Returns whether an edge is contested by the opponent."""
        neighbors = self.neighboring_hexes(edge)
        for neighbor in neighbors:
            piece = self.get_piece(neighbor)
            if piece != None and piece.player != player:
                return True
        return False

    def legal_placement_moves(self, piece, edge):
        """Returns all legal placement moves for a piece."""
        moves = []
        # add all possible placement moves. A piece cannot be placed next to an opponent's piece
        for e in edge:
            # check if the edge of e does not contain an opponent's piece
            if not self.check_contested(e, piece.player) or self.game.turn_number == 1:
                moves.append(Move(piece.player, piece, e))
        return moves

class Move:
    """Represents a move on the board."""
    
    def __init__(self, player, piece, new_position):
        self.piece = piece
        self.player = player
        self.old_position = piece.position
        self.old_edge = piece.edge
        self.position = new_position
        self.edge = self.generate_new_edge()

    def generate_new_edge(self):
        """Generates the edge of a piece after the move."""
        reach = [[self.position[i] + adj[i] for i in range(len(self.position))] for adj in Piece.adjacent] 
        new_edge = [None for i in range(len(reach))]
        for i, reachable in enumerate(reach):
            piece = self.player.active_game.board.get_piece(reachable)
            if piece != None and reachable != self.old_position:
                new_edge[i] = piece

        return new_edge

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.piece, self.position)

class Player:
    """Represents a player in the game."""
    
    def __init__(self, name):
        self.name = name
        self.active_game = None

    def play_game(self, game):
        """Enters the player into a game."""
        self.active_game = game

    def is_Whiteside(self):
        """Returns true if the player is playing as the whiteside."""
        return self.isWhite

    def make_move(self, move):
        """Makes a move on the board.
        
        #### Raises:
        RuntimeError: if the player is not in a game.
        """
        if self.active_game == None:
            raise Exception("Player is not playing a game")
        self.active_game.make_move(move)

class HumanPlayer(Player):
    
    def __repr__(self) -> str:
        return self.name

class ComputerPlayer(Player):
    
    def __repr__(self):
        return "{}".format(self.__class__.__name__)
    
    def make_move(self):

        possible_moves = self.active_game.all_legal_moves()

        # choose a random move
        move = random.choice(possible_moves)

        # make the move
        super().make_move(move)

class Piece:

    # six hexagonal neighbors
    adjacent = [(1,1,0), (0,1,1), (-1,0,1), (-1,-1,0), (0,-1,-1), (1,0,-1)]

    def __init__(self, player, game):
        self.position = None
        self.edge = None
        self.player = player
        self.mounted = False
        self.game = game

    def update_neighbors(self):
        """Updates the edge of the piece based on the current board state."""
        neighbors = [piece for piece in self.edge if piece != None]
        for piece in neighbors:
            for i, adj in enumerate(Piece.adjacent):
                if self.position == [piece.position[i] + adj[i] for i in range(len(piece.position))]:
                    piece.edge[i] = self
                    break

    def remove_from_old_neighbors(self):
        """Remove the piece from the edge of its old neighbors."""
        if self.position is not None:
            neighbors = [piece for piece in self.edge if piece != None]
            for piece in neighbors:
                for i, adj in enumerate(Piece.adjacent):
                    if self.position == [piece.position[i] + adj[i] for i in range(len(piece.position))]:
                        piece.edge[i] = None
                        break

    def make_move(self, move):
        """Makes a move on the board.
        
        removes the piece from its old neighbors and updates the new neighbors.
        """
        self.remove_from_old_neighbors()

        self.position = move.position
        self.edge = move.edge

        self.update_neighbors()

    def is_legal_move(self, move):
        """Try to make a piece-independent move.
        
        #### Returns:
        True if the move is legal, False otherwise.

        This should be called by the try_move method of the piece to check the legality of the move before checking 
        piece-specific rules.
        """

        # make sure the piece is not on top of another piece
        for piece in self.game.board.pieces: 
            if piece.position == move.position:
                print("piece is on top of another piece")
                return False
            
        # check if new position is connected to the hive 
        first_piece = self.game.turn_number == 1 and self.game.turn == self.game.player1 
        if move.edge == [None for i in range(len(move.edge))] and not first_piece:
            print("piece is not connected to the hive")
            return False

        # check if the placed piece is not touching an opponents piece
        if move.old_position == None and self.game.turn_number > 1:
            neighboring_players = set([piece.player for piece in move.edge if piece != None])
            if len(neighboring_players) == 1 and move.piece.player in neighboring_players:
                return True
            else:
                print("piece is touching an opposing piece")
                return False

        # check if the hive is still connected
        # since we already checked that the new position is connected, it is enough to check if the hive without the moving piece is connnected
        if not self.game.board.is_connected(move):
            print("hive is not connected")
            return False

        return True
    
    def try_move(self, move):
        if Piece.is_legal_move(self, move):
            if self.position == None: 
                return True
            if self.is_legal_move(move.position, move.edge):
                return True
        return False
        
    def num_neighbors(self):
        """Returns the number of neighbors of the piece."""
        if self.edge == None:
            return 0
        return len([piece for piece in self.edge if piece != None])

    def reachable(self, position, edge=None):
        """Returns the reachable positions from a given position.
        
        Return all the reachable positions taking into account that moving through narrow passages is not allowed.
        """
        if edge==None:
            edge = self.game.board.get_edge(position)

        reach = [[position[i] + adj[i] for i in range(len(position))] for adj in Piece.adjacent] 
        rm = []
        for i, hex in enumerate(reach):
            if edge[i] != None or ((edge[(i+1)%6] != None) == (edge[(i-1)%6] != None)):
                rm.append(i)
                continue
            if edge[(i+1)%6] != None and edge[(i-1)%6] != None:
                rm.append(i)
        for i in reversed(rm):
            reach.pop(i)

        return reach

    def legal_moves(self, edges, movement="Allowed"):
        """Returns all the legal moves of the piece.
        
        #### Arguments:
        edges: coordinates of the edges of the board.
        movement: "Place bee" if the bee has to be placed, "Illegal" if only placement is allowed, "Allowed" if all moves are allowed.
        """
        moves = []

        if movement == "Place bee":
            if self.__str__() == "Bee":
                moves += self.game.board.legal_placement_moves(self, edges)
            return moves

        if self.mounted == True:
            return moves

        if self.position == None:
            moves += self.game.board.legal_placement_moves(self, edges)
            return moves 

        if movement == "Illegal":
            return moves
        
        moves += self.legal_movement(edges)
        return moves
    
    def legal_movement(self, edges):
        """Returns all the legal moves of the piece.
        
        #### Arguments:
        edges: coordinates of the edges of the board.
        """
        moves = []
        
        for i, edge in enumerate(edges):
            move = Move(self.player, self, edge)
            if self.try_move(move):
                moves.append(move)
        return moves    

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.player, self.position)

    def __str__(self):
        return "{}".format(self.__class__.__name__)

class Bee(Piece):

    def __init__(self, player, game):
        super().__init__(player, game)
        self.placed = False

    def is_legal_move(self, new_position, new_edge):
        if self.num_neighbors() == 5:
            return False
        reachable_edges = self.reachable(self.position, self.edge)
        if new_position in reachable_edges:
            return True
        return False

    def try_move(self, move):
        if Piece.is_legal_move(self, move):
            if self.position == None: 
                self.placed = True
                return True
            if self.is_legal_move(move.position, move.edge):
                return True
        return False

class Beetle(Piece):

    def __init__(self, player, game):
        super().__init__(player, game)
        self.mounting = False
        self.mounted_piece = None

    def return_mounted_piece(self):
        self.mounted_piece.mounted = False
        self.mounted_piece.position = self.position
        self.mounted_piece.edge = self.edge
        self.mounted_piece.update_neighbors()
        self.mounted_piece = None

    def try_move(self, move):

        # make sure the piece is not on top of another piece when placed
        if self.position == None:
            for piece in self.game.board.pieces: 
                if piece.position == move.position:
                    print("piece is on top of another piece")
                    return False

        # check if new position is connected to the hive 
        first_piece = self.game.turn_number == 1 and self.game.turn == self.game.player1 
        if move.edge == [None for i in range(len(move.edge))] and not first_piece and self.mounting == False:
            print("piece is not connected to the hive")
            return False

        # check if the placed piece is not touching an opponents piece
        if move.old_position == None and self.game.turn_number > 1:
            neighboring_players = set([piece.player for piece in move.edge if piece != None])
            if len(neighboring_players) == 1 and move.piece.player in neighboring_players:
                return True
            else:
                print("piece is touching an opposing piece")
                return False

        # check if the hive is still connected
        # since we already checked that the new position is connected, it is enough to check if the hive without the moving piece is connnected
        if not self.mounting and not self.game.board.is_connected(move):
            print("hive is not connected")
            return False

        if self.position == None: 
            return True
        if self.is_legal_move(move.position, move.edge):
            return True
        return False

    def make_move(self, move):
        if self.position != None:
            neighbors = [piece.position for piece in self.edge if piece != None]
            if move.position in neighbors:
                # now the mounted piece will temporarily be removed from the game
                if self.mounting:
                    self.return_mounted_piece()
                self.mounting = True
            else:
                if self.mounting:
                    # self.return_mounted_piece()
                    self.mounting = False

        self.remove_from_old_neighbors()

        # if the beetle is mounting a piece, return the mounted piece
        if self.mounted_piece != None:
            self.return_mounted_piece()

        if self.mounting:
            self.mounted_piece = self.player.active_game.board.get_piece(move.position)
            self.mounted_piece.position = None
            self.mounted_piece.edge = None
            self.mounted_piece.mounted = True
        
        self.position = move.position
        self.edge = self.game.board.get_edge(move.position)

        self.update_neighbors()
        # also update next-nearest neighbors
        for piece in self.edge:
            if piece != None:
                piece.update_neighbors()

    def is_legal_move(self, new_position, new_edge):
        if self.mounting:
            reachable_edges = [[self.position[i] + adj[i] for i in range(len(self.position))] for adj in Piece.adjacent] 
        else:
            reachable_edges = self.reachable(self.position, self.edge)
        neighbors = [piece.position for piece in self.edge if piece != None]
        if new_position in neighbors:
            return True
        if new_position in reachable_edges:
            return True
        return False

    def legal_movement(self, edges):
        potential_moves = edges
        for n in self.edge:
            if n != None:
                potential_moves.append(n.position)

        moves = super().legal_movement(potential_moves)

        return moves

class Ant(Piece):
    
    def is_legal_move(self, new_position, new_edge):
        if self.num_neighbors() == 5:
            return False
        visited = []
        stack = [self.position]
        while len(stack) > 0:
            position = stack.pop()
            visited.append(position)
            edge = self.game.board.get_edge(position)
            reachable_edges = self.reachable(position, edge)
            for adj in reachable_edges:
                if adj not in visited:
                    stack.append(adj)
                
        if new_position in visited:
            return True
        return False

class Grasshopper(Piece):

    # the grasshoper can only move by jumping over pieces in a straight line
    def is_legal_move(self, new_position, new_edge):
        direction = [new_position[i] - self.position[i] for i in range(len(self.position))]
        distance = abs(max(direction, key=abs))
        direction = [direction[i]/distance for i in range(len(direction))]
        
        # if the direction does not contain a 0, then the grasshopper is not moving in a straight line and is therefore not a legal move
        if not (direction.count(0) == 1):
            print("not a straight line")
            return False

        # if the distance is not greater than 1, then the grasshopper is not jumping over any pieces and is therefore not a legal move
        if distance < 2:
            print("not jumping over any pieces")
            return False
        
        # check if all positions between the current position and the new position are filled
        for i in range(1, distance):
            if self.game.board.get_piece([self.position[j] + direction[j]*i for j in range(len(self.position))]) == None:
                print("there are some gaps")
                return False
        
        return True

    # def try_move(self, move):
    #     if super().try_move(move):
    #         if self.position == None: 
    #             self.make_move(move)
    #             return True
    #         if self.is_legal_move(move.position, move.edge):
    #             self.make_move(move)
    #             return True
    #     return False

class Spider(Piece):
    """
    The spider is a piece that can move exactly 3 spaces along the edge of the hive
    """

    def is_legal_move(self, new_position, new_edge):
        # if the spider has 5 neighbors, it cannot move
        if self.num_neighbors() == 5:
            print("spider has 5 neighbors")
            return False

        # get positions that are exactly 3 spaces away from the current position
        reachable1 = self.reachable(self.position, self.edge)
        reachable2 = [r for p in reachable1 for r in self.reachable(p) if r != self.position] 
        reachable3 = [r for p in reachable2 for r in self.reachable(p) if r not in reachable1]

        # if the new position is one of the reachable positions, then the move is legal
        if new_position in reachable3:
            return True

        print("new position is not reachable")
        return False


if __name__ == "__main__":
    Gregor = HumanPlayer("Gregor")
    Wilke = HumanPlayer("Wilke")
    game = Game(Gregor, Wilke)

    firstMove = Move(Gregor, game.board.pieces[0], [0,0,0])
    game.make_move(firstMove)
    print(len(game.all_legal_moves()))
    secondMove = Move(Wilke, game.board.pieces[9], [1,1,0])
    game.make_move(secondMove)
    thirdMove = Move(Gregor, game.board.pieces[0], [0,1,1])
    game.make_move(thirdMove)

    gregs_queen = [piece for piece in game.board.pieces if piece.player == Gregor][0]
    # print(game.all_legal_moves())
