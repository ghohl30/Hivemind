import random
from typing import Optional
import warnings
import numpy as np

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

        # currently legal moves
        self.legal_moves = self.all_legal_moves()

    def make_move(self, move):
        """Make a move on the board.

        Passes the move to the board, and then updates whos turn it is
        
        Parameters:
        ------------
        move : Move
            The move to be made.
        """

        # if self.winner is instance of Player, then the game is over
        if isinstance(self.winner, Player):
            print(self.winner)
            warnings.warn("Game is over")   
            # return the name of the winner
            return self.winner.name
        
        # for the first for turns check:
        # 1. if the bee has not been placed the move is a placement move
        # 2. if the bee has not been placed within the first 3 moves decline any non-placement moves for the 4th turn
        if self.turn_number < 5:
            bee = self.board.get_bee(move.player)
            if move.piece.position is not None:
                # if bee.position is None:
                if not bee.placed:
                    warnings.warn("You cannot move before placing the bee")
                    return False
            if self.turn_number == 4:
                if not bee.placed and move.piece.__str__() != "QueenBee":
                    warnings.warn("You have to place your bee within the first 4 turns")
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
            # print(self.turn)
            if self.turn == self.player1:
                self.turn_number += 1
        
        # update legal moves
        self.legal_moves = self.all_legal_moves()
        if len(self.legal_moves) == 0:
            self.winner = self.player2 if self.turn == self.player1 else self.player1
            print("Game Over")
        return res
    
    def matrix_to_move(self, matrix):
        # matrix is a 22x22x7 np array

        if np.sum(matrix[:,:,0])!=0:
            #this is the first move
            # TODO
            pass

        # get the index of max value
        mov_index = np.unravel_index(matrix.argmax(), matrix.shape)

        # get the piece
        piece = self.board.get_piece_from_label(Board.labels[mov_index[0]])
        
        # get the position of known new neighbour
        neighbor = self.board.get_piece_from_label(Board.labels[mov_index[1]])

        pos = [neighbor.position[i] + Piece.adjacent[mov_index[2]-1][i] for i in range(3)]
        move = Move(piece.player, piece, pos)

        return move

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
                warnings.warn("You have to place your bee within the first 4 turns")
            

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
    game_mode = [('QueenBee', 1), ('Ant',3), ('Beetle', 2), ('Spider', 2), ('Grasshopper', 3)]
    # game_mode = [('Bee', 1), ('Beetle', 2)]

    # List of labels
    labels = ['wQ1', 'wA1', 'wA2', 'wA3', 'wG1', 'wG2', 'wG3', 'wB1', 'wB2', 'wS1', 'wS2', 
                'bQ1', 'bA1', 'bA2', 'bA3', 'bG1', 'bG2', 'bG3', 'bB1', 'bB2', 'bS1', 'bS2']
    
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
                if p[0] == 'QueenBee':
                    for i in range(p[1]):
                        self.pieces.append(Bee(player, self.game, i+1))
                elif p[0] == 'Ant':
                    for i in range(p[1]):
                        self.pieces.append(Ant(player, self.game, i+1))
                elif p[0] == 'Beetle':
                    for i in range(p[1]):
                        self.pieces.append(Beetle(player, self.game, i+1))
                elif p[0] == 'Spider':
                    for i in range(p[1]):
                        self.pieces.append(Spider(player, self.game, i+1))
                elif p[0] == 'Grasshopper':
                    for i in range(p[1]):
                        self.pieces.append(Grasshopper(player, self.game, i+1))

    # returns the bee object of the given player
    def get_bee(self, player):
        """Returns the bee of the given player."""
        for piece in self.pieces:
            if piece.__str__() == 'QueenBee' and piece.player == player:
                return piece
        raise RuntimeError("No bee found for this player")

    def get_piece_from_label(self, name):
        """Returns the piece with the given label."""
        for piece in self.pieces:
            if piece.name == name:
                return piece
        raise RuntimeError("No piece found with this name")

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

        bee1 = self.get_bee(self.game.player1)
        bee2 = self.get_bee(self.game.player2)

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
            warnings.warn("piece is the first piece")
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
    
    def move_mask(self):
        """Returns a mask of all possible moves."""
        mask = np.zeros((22,22,7))

        # if it is the first turn layers 1-6 stay 0. Layer 0 is 1 on the diagonal until 11x11
        if self.game.turn_number == 1 and self.game.turn == self.game.player1:
            for i in range(11):
                mask[i,i,0] = 1
                return mask

        # layer 0 stays 0. The rest is determined from the legal moves of the game class
        for move in self.game.legal_moves:
            idxs = [(self.labels.index(move.piece.name), self.labels.index(p.name), ((k+3)%6)+1) for k, p in enumerate(move.edge) if p is not None]
            for idx in idxs:
                mask[idx] = 1/len(idxs)

        return mask
    
    def get_matrices(self):
        
        # Adjacency matrix
        adj = np.zeros((22,22))

        # Mounting matrix
        mount = np.zeros((22,22))

        # Set firstPiece
        if self.game.turn == self.game.player2 and self.game.turn_number == 1:
            firstPiece = self.pieces[0]
            firstPieceIdx = self.labels.index(firstPiece.name)
            firstPiece_layer = np.ones((22,22)) * firstPieceIdx
        else:
            firstPiece_layer = np.zeros((22,22))

        for piece in self.pieces:
            if piece.position is not None:
                index = self.labels.index(piece.name)
                adj[index, index] = -1
                for i, neighbor in enumerate(piece.edge):
                    if neighbor is not None:
                        neighbor_index = self.labels.index(neighbor.name)
                        adj[index, neighbor_index] = i+1
                # if the piece is a beetle, check if it is mounting. If so add the mounted piece to the adjacency matrix
                if piece.__str__() == 'Beetle' and piece.mounting:
                    mounted_piece = piece.mounted_piece
                    mounted_index = self.labels.index(mounted_piece.name)
                    
                    mount[index, mounted_index] = 1
                    mount[mounted_index, index] = -1

                    if mounted_piece.__str__() == 'Beetle' and mounted_piece.mounting:
                        mounted_piece = mounted_piece.mounted_piece
                        mounted_index = self.labels.index(mounted_piece.name)
                        
                        mount[index, mounted_index] = 1
                        mount[mounted_index, index] = -1

                        if mounted_piece.__str__() == 'Beetle' and mounted_piece.mounting:
                            mounted_piece = mounted_piece.mounted_piece
                            mounted_index = self.labels.index(mounted_piece.name)
                            
                            mount[index, mounted_index] = 1
                            mount[mounted_index, index] = -1

        return adj, mount, firstPiece_layer

    def get_graph(self):
        # get feature vector x, edge_index and edge attribute

        x = np.zeros((22, 1))

        # # get edge_index and edge_attr
        edge_index = []
        edge_attr = []

        for piece in self.pieces:
            if piece.position is not None:
                index = self.labels.index(piece.name)
                x[index] = 1
                for i, neighbor in enumerate(piece.edge):
                    if neighbor is not None:
                        neighbor_index = self.labels.index(neighbor.name)
                        edge_index.append([index, neighbor_index])
                        edge_attr.append(i+1)
                # if the piece is a beetle, check if it is mounting. If so add the mounted piece to the adjacency matrix
                if piece.__str__() == 'Beetle' and piece.mounting:
                    mounted_piece = piece.mounted_piece
                    mounted_index = self.labels.index(mounted_piece.name)

                    x[index] = 2

                    if mounted_piece.__str__() == 'Beetle' and mounted_piece.mounting:
                        mounted_piece = mounted_piece.mounted_piece
                        mounted_index = self.labels.index(mounted_piece.name)

                        x[index] = 3

                        if mounted_piece.__str__() == 'Beetle' and mounted_piece.mounting:
                            x[index] = 4

        edge_index = np.array(edge_index).T
        edge_attr = np.array(edge_attr)

        return x, edge_index, edge_attr




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
    
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def play_game(self, game: Game):
        """Enters the player into a game."""
        self.active_game = game

    def make_move(self, move):
        """Makes a move on the board.
        
        #### Raises:
        RuntimeError: if the player is not in a game.
        """
        if self.active_game == None:
            raise Exception("Player is not playing a game")
        self.active_game.make_move(move)

    def __repr__(self):
        return self.name
    
class HumanPlayer(Player):
    
    def __repr__(self) -> str:
        return self.name

class ComputerPlayer(Player):
    
    def __repr__(self):
        return self.name
    
    def make_move(self):

        possible_moves = self.active_game.legal_moves

        # choose a random move
        move = random.choice(possible_moves)

        # make the move
        super().make_move(move)

    def make_random_move(self):

        possible_moves = self.active_game.legal_moves

        # choose a random move
        move = random.choice(possible_moves)

        # make the move
        super().make_move(move)

class Piece:

    # six hexagonal neighbors
    adjacent = [(1,1,0), (0,1,1), (-1,0,1), (-1,-1,0), (0,-1,-1), (1,0,-1)]

    def __init__(self, player, game, id):
        self.name = player.color + self.__str__()[0] + str(id)
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
                warnings.warn("piece is on top of another piece")
                return False
            
        # check if new position is connected to the hive 
        first_piece = self.game.turn_number == 1 and self.game.turn == self.game.player1 
        if move.edge == [None for i in range(len(move.edge))] and not first_piece:
            warnings.warn("piece is not connected to the hive")
            return False

        # check if the placed piece is not touching an opponents piece
        if move.old_position == None and self.game.turn_number > 1:
            neighboring_players = set([piece.player for piece in move.edge if piece != None])
            if len(neighboring_players) == 1 and move.piece.player in neighboring_players:
                return True
            else:
                warnings.warn("piece is touching an opposing piece")
                return False

        # check if the hive is still connected
        # since we already checked that the new position is connected, it is enough to check if the hive without the moving piece is connnected
        if not self.game.board.is_connected(move):
            warnings.warn("hive is not connected")
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

        # replace appearances of self in edge with None
        edge = [None if piece == self else piece for piece in edge]

        reach = [[position[i] + adj[i] for i in range(len(position))] for adj in Piece.adjacent] 
        neighbors_reach = [[piece.position[i] + adj[i] for i in range(len(piece.position))] for piece in edge if piece != None for adj in Piece.adjacent]
        rm = []
        for i, hex in enumerate(reach):
            # check for narrow passages
            if edge[i] != None or ((edge[(i+1)%6] != None) and (edge[(i-1)%6] != None)):
                rm.append(i)
                continue
            # check if new position would share a neighbor with the current position of the piece

        for i in reversed(rm):
            reach.pop(i)
 
        reach = [pos for pos in reach if pos in neighbors_reach]
        return reach

    def legal_moves(self, edges, movement="Allowed"):
        """Returns all the legal moves of the piece.
        
        #### Arguments:
        edges: coordinates of the edges of the board.
        movement: "Place bee" if the bee has to be placed, "Illegal" if only placement is allowed, "Allowed" if all moves are allowed.
        """
        moves = []

        if movement == "Place bee":
            if self.__str__() == "QueenBee":
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

    def __init__(self, player, game, id):
        super().__init__(player, game, id)
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

    def __str__(self):
        return "QueenBee"

class Beetle(Piece):

    def __init__(self, player, game, id):
        super().__init__(player, game, id)
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
                    warnings.warn("piece is on top of another piece")
                    return False

        # check if new position is connected to the hive 
        first_piece = self.game.turn_number == 1 and self.game.turn == self.game.player1 
        if move.edge == [None for i in range(len(move.edge))] and not first_piece and self.mounting == False:
            warnings.warn("piece is not connected to the hive")
            return False

        # check if the placed piece is not touching an opponents piece
        if move.old_position == None and self.game.turn_number > 1:
            neighboring_players = set([piece.player for piece in move.edge if piece != None])
            if len(neighboring_players) == 1 and move.piece.player in neighboring_players:
                return True
            else:
                warnings.warn("piece is touching an opposing piece")
                return False

        # check if the hive is still connected
        # since we already checked that the new position is connected, it is enough to check if the hive without the moving piece is connnected
        if not self.mounting and not self.game.board.is_connected(move):
            warnings.warn("hive is not connected")
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
            warnings.warn("not a straight line")
            return False

        # if the distance is not greater than 1, then the grasshopper is not jumping over any pieces and is therefore not a legal move
        if distance < 2:
            warnings.warn("not jumping over any pieces")
            return False
        
        # check if all positions between the current position and the new position are filled
        for i in range(1, distance):
            if self.game.board.get_piece([self.position[j] + direction[j]*i for j in range(len(self.position))]) == None:
                warnings.warn("there are some gaps")
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
            warnings.warn("spider has 5 neighbors")
            return False

        # get positions that are exactly 3 spaces away from the current position
        reachable1 = self.reachable(self.position, self.edge)
        reachable2 = [r for p in reachable1 for r in self.reachable(p) if r != self.position] 
        reachable3 = [r for p in reachable2 for r in self.reachable(p) if r not in reachable1]

        # if the new position is one of the reachable positions, then the move is legal
        if new_position in reachable3:
            return True

        warnings.warn("new position is not reachable")
        return False


if __name__ == "__main__":
    Gregor = HumanPlayer("Gregor", 'w')
    Wilke = HumanPlayer("Wilke", 'b')
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
