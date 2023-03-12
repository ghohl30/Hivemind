class Piece:
    def try_move(self):
        # implementation of try_move for Piece class
        pass

    def new_method(self):
        # implementation of new_method for Piece class
        Piece.try_move(self)
        # other code for new_method

class Spider(Piece):
    def try_move(self):
        # implementation of try_move for Spider class
        pass

# create a Piece object
p = Piece()

# call new_method on the Piece object
p.new_method()