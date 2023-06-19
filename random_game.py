import hive
import utils
import matplotlib.pyplot as plt
import imageio
import os

from ComputerPlayers import AgressiveComputer

# Create a subfolder named 'img'
if not os.path.exists('img'):
    os.makedirs('img')

# player1 = hive.ComputerPlayer("Gregor", "w")
player1 = AgressiveComputer("Agressive", "w")
player2 = hive.ComputerPlayer("Wilke", "b")

game = hive.Game(player1, player2)

game_over = False
move_num = 0
filenames = []  # Store all image file names

while not game_over:
    game.turn.make_move()
    if game.winner is not None:
        game_over = True
        print("The winner is {}!".format(game.winner))
    move_num += 1
    print("Move number: {}".format(move_num))

    A, M, firstPiece = game.board.get_matrices()
    
    # Plot hexagons and save the plot as an image
    utils.plot_hexagons(A, M, firstPiece)
    filename = f'img/move_{move_num}.png'
    plt.savefig(filename)
    filenames.append(filename)  # Add filename to list
    plt.close()  # Close the plot


# Convert images into a GIF
with imageio.get_writer('game.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image) # type: ignore

# List of images
images = [imageio.imread(filename) for filename in filenames]

# Save the images as a video
imageio.mimsave('game.mp4', images, fps=5) # type: ignore

# Remove all image files
for filename in filenames:
    os.remove(filename)
