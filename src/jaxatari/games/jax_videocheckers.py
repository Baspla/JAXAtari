import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame

from jaxatari.environment import JAXAtariAction as Action

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

#
# by Tim Morgner and Jan Larionow
#

# Note
# For us, play.py does not work consistently and sometimes freezes. However, based on the Mattermost
# messages, we assume that it works for you. We have not modified play.py.

# region Constants
# Constants for game environment
WIDTH = 160
HEIGHT = 210
# endregion

# region Pygame Constants
# Pygame window dimensions
SCALING_FACTOR = 3
WINDOW_WIDTH = WIDTH * SCALING_FACTOR
WINDOW_HEIGHT = HEIGHT * SCALING_FACTOR
# endregion

# region Offsets
OFFSET_X_BOARD = 12
OFFSET_Y_BOARD = 50
# endregion

# region moves
# in (y, x) notation
MOVES = jnp.array([
    [1, 1],  # UPRIGHT
    [-1, 1],  # DOWNRIGHT
    [-1, -1],  # DOWNLEFT
    [1, -1],  # UPLEFT
])
# endregion

# region Checker pieces
EMPTY_TILE = 0
WHITE_PIECE = 1
BLACK_PIECE = 2
WHITE_KING = 3
BLACK_KING = 4
# endregion

class VideoCheckersState(NamedTuple):
    cursor_pos: chex.Array
    board: chex.Array # Shape (32,1) for 8x8 board only black fields and 2 dimension for piece type.

    turn: chex.Array
    selected_piece: chex.Array
    destination: chex.Array
    jump_available: chex.Array

    animation_frame: chex.Array

class VideoCheckersObservation(NamedTuple):
    board: chex.Array # All animation is already baked into the board observation
    start_pos: chex.Array
    end_pos: chex.Array

class VideoCheckersInfo(NamedTuple):
    all_rewards: chex.Array

#TODO: adjust logic to work with "we can't afford a real matrix" approach
@partial(jax.jit, static_argnums=(0,))
def move_step(move_action, state: VideoCheckersState) -> VideoCheckersState:
    """
    Handles the movement of the cursor on the board.
    Args:
        move_action: either UPRIGHT, UPLEFT, DOWNLEFT or DOWNRIGHT
        state: Game state

    Returns: The new game state
    """
    # get direction components
    up = jnp.logical_or(move_action == Action.UPLEFT, move_action == Action.UPRIGHT)
    right = jnp.logical_or(move_action == Action.DOWNRIGHT, move_action == Action.UPRIGHT)

    dy = jax.lax.cond(up, lambda s: 1, lambda s: -1, None)
    dx = jax.lax.cond(right, lambda s: 1, lambda s: -1, None)

    def handle_move_no_selection(dx, dy, state: VideoCheckersState):
        new_x = state.cursor_pos[0] + dx
        new_y = state.cursor_pos[1] + dy
        return jax.lax.cond(move_in_bounds(dx, dy, state),
                            lambda s: s._replace(cursor_x=new_x, cursor_y=new_y),
                            lambda s: s,
                            operand=state)

    def handle_move_with_selection(dx, dy, state: VideoCheckersState):
        """
        Handles the move input for when a piece was selected. Enforces the rule that a jump must be taken, if it is
        available. E.g., if a piece has two moves available, jump to up-right and move to up-left, it will be barred
        from taking the move to the top left. Any attempts at making the move will result in an unchanged state.
        Args:
            dx: movement in x direction
            dy: movement in y direction
            state: state of the game

        Returns: New state after applying the move.

        """
        possible_moves = get_possible_moves_for_piece(state.selected_piece[0], state.selected_piece[0], state)

        # check if given move is in possible moves (either as simple or as jump)
        attempted_jump = jnp.array([dx * 2, dy * 2])
        attempted_normal_move = jnp.array([dx, dy])

        # compares rowwise all possible moves to input move. Returns true if for any, both movement components match
        is_valid_jump = jnp.any(jnp.all(possible_moves == attempted_jump, axis=1))
        is_valid_normal_move = jnp.any(jnp.all(possible_moves == attempted_normal_move, axis=1))

        # enforces that a jump is taken, if one is available
        move_is_valid = jax.lax.cond(
            state.jump_available,
            lambda s: is_valid_jump,
            lambda s: is_valid_normal_move | is_valid_jump,
            operand=None
        )

        final_dx = jax.lax.cond(is_valid_jump, lambda: dx * 2, lambda: dx)
        final_dy = jax.lax.cond(is_valid_jump, lambda: dy * 2, lambda: dy)

        return jax.lax.cond(
            move_is_valid,
            lambda s: s._replace(cursor_x=s.cursor_x + final_dx, cursor_y=s.cursor_y + final_dy),
            # move cursor if move is valid
            lambda s: s,  # dont do anything if not
            operand=state
        )

    has_selection = (state.selected_piece[0] != -1) & (state.selected_piece[0] != -1)
    return jax.lax.cond(
        has_selection,
        handle_move_with_selection,
        handle_move_no_selection,
        operand=[dx, dy, state],
    )


@partial(jax.jit, static_argnums=(0,))
def get_possible_moves_for_piece(x, y, state: VideoCheckersState):
    """
    Get all possible moves for a piece at position (y,x)
    Args:
        x: x coordinate of piece
        y: y coordinate of piece
        state: current game state

    Returns: array of all possible moves.
    """

    current_piece = state.board[y, x]
    dy_forward = jax.lax.cond(state.turn == WHITE_PIECE, lambda s: 1, lambda s: -1, operand=None)
    piece_is_king = (current_piece == WHITE_KING) | (current_piece == BLACK_KING)

    def check_move(move):
        dy, dx = move

        is_forward = (dy == dy_forward)
        can_move_in_direction = jax.lax.cond(piece_is_king, lambda s: True, lambda s: is_forward, operand=None)

        def get_valid_moves_for_direction():
            jump_available = move_is_available(2 * dx, 2 * dy, state)  # check jump
            move_available = move_is_available(dx, dy, state)  # check normal move

            # Return jump move if available, else normal move if available, else [0,0]
            return jax.lax.cond(
                jump_available,
                lambda s: move * 2,
                lambda s: jax.lax.cond(
                    move_available,
                    lambda s: move,
                    lambda s: jnp.array([0, 0]),
                    operand=None),
                operand=None
            )

        return jax.lax.cond(can_move_in_direction,
                            get_valid_moves_for_direction,
                            lambda s: jnp.array([0, 0]),
                            operand=None)

    possible_moves = jax.vmap(check_move)(MOVES)

    # Filter out zero moves ([0,0]) - keep only valid moves
    valid_moves = possible_moves[jnp.any(possible_moves != 0, axis=1)]

    return valid_moves


@partial(jax.jit, static_argnums=(0,))
def move_in_bounds(dx, dy, state: VideoCheckersState):
    """
    Checks if cursor can be moved in the given direction.
    Args:
        dx: movement in x direction
        dy: movement in y direction
        state: state of the game, containing current cursor position.

    Returns: True, if cursor can be moved in the given direction, False otherwise.

    """
    return ((state.cursor_pos[1] + dy >= 0) & (state.cursor_pos[1] + dy < NUM_FIELDS_Y) &
            (state.cursor_pos[0] + dx >= 0) & (state.cursor_pos[0] + dx < NUM_FIELDS_X))


@partial(jax.jit, static_argnums=(0,))
def move_is_available(dx, dy, state: VideoCheckersState):
    """
    Checks if a piece can be moved in the given direction. Checks for both, simple moves and jumps.
    Args:
        dx: movement in x direction
        dy: movement in y direction
        state: state of the game, containing position of the current piece and the board-state.

    Returns:
        True, if a piece can be moved in the given direction, False otherwise.
    """
    landing_in_bounds = move_in_bounds(dx, dy, state)
    x, y, board = state.cursor_pos[0], state.cursor_pos[1], state.board

    def handle_jump():
        """
        Handle moves with |dx|=2 and |dy|=2
        Returns: True if that movement is available, False otherwise.
        """
        own_colour = state.board[y, x]
        jumped_x = x + (dx // 2)
        jumped_y = y + (dy // 2)
        return jax.lax.cond(
            landing_in_bounds,
            lambda s: (board[jumped_y, jumped_x] != EMPTY_TILE) &  # jumped-tile is not empty
                      (board[jumped_y, jumped_x] != own_colour) &  # jumped-tile is not of same colour
                      (board[y + 2 * dy, x + 2 * dx] == EMPTY_TILE),  # landing tile is empty
            lambda s: False,
            operand=None
        )

    def handle_move():
        """
        Handle moves with |dx|=1 and |dy|=1
        Returns: True if that movement is available, False otherwise.
        """
        return landing_in_bounds and (board[y + dy, x + dx] == EMPTY_TILE)

    is_jump = (jnp.abs(dx) == 2) & (jnp.abs(dy) == 2)
    return jax.lax.cond(is_jump, handle_jump, handle_move)


@partial(jax.jit, static_argnums=(0,))  # TODO
def select_tile(select_action, state: VideoCheckersState) -> VideoCheckersState:
    """
    no selection, jump available, own piece with jump -> update state
    no selection, jump available, own piece w/o jump -> nothing
    no selection, no jump available, own piece with jump -> update state
    no selection, no jump available, own piece w/o jump -> update state

    selection, jump available, own piece with jump -> update state
    selection, jump available, own piece w/o jump -> nothing
    selection, no jump available, own piece with jump -> update state
    selection, no jump available, own piece w/o jump -> update state

    empty or enemy piece -> nothing
    """


class JaxVideoCheckers(JaxEnvironment[VideoCheckersState, VideoCheckersObservation, VideoCheckersInfo]):
    def __init__(self, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            Action.FIRE,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT
        }
        # TODO: Nachfragen ob NOOP Action benötigt wird

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(187)) -> Tuple[
        VideoCheckersObservation, VideoCheckersState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)

        Args:
            key: Random key for generating the initial state.
        Returns:
            initial_obs: Initial observation of the game.
            state: Initial game state.
        """
        # Initialize the board with pieces, this is a placeholder
        # Pieces are 0 for empty, 1 for player 1 pieces, 2 for player 2 pieces, 3 for player 1 kings, 4 for player 2 kings.
        board = jnp.zeros(32, dtype=jnp.int32)
        # Set up the initial pieces on the board
        # Player 1 pieces
        board = board.at[0:8].set(1)
        # Player 2 pieces
        board = board.at[24:32].set(2)
        state = VideoCheckersState(cursor_pos=jnp.array([0, 0]), board= board,
                                   selected_piece=jnp.array([-1, -1]), animation_frame=jnp.array(0))


        initial_obs = self._get_observation(state)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: VideoCheckersState):
        """
        Returns the observation of the game state.
        Args:
            state: The current game state.
        Returns:
            VideoCheckersObservation: The observation of the game state.
        """
        return VideoCheckersObservation(board=state.board,
                                        start_pos=state.cursor_pos,
                                        end_pos=state.selected_piece)
        #TODO generate valid observation instead of placeholder

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: VideoCheckersState, action: chex.Array) -> Tuple[
        VideoCheckersObservation, VideoCheckersState, float, bool, VideoCheckersInfo]:
        """
        Takes a step in the game environment based on the action taken.
        Args:
            state: The current game state.
            action: The action taken by the player.
        Returns:
            observation: The new observation of the game state.
            new_state: The new game state after taking the action.
            reward: The reward received after taking the action.
            done: A boolean indicating if the game is over.
            info: Additional information about the game state.
        """

        new_state = VideoCheckersState(
            cursor_pos=state.cursor_pos,
            board=state.board,
            selected_piece=state.selected_piece,
            animation_frame=state.animation_frame
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def action_space(self):
        """
        Returns the action space of the game environment.
        Returns:
            action_space: The action space of the game environment.
        """
        return jnp.array(list(self.action_set), dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: VideoCheckersState, all_rewards: chex.Array) -> VideoCheckersInfo:
        """
        Returns additional information about the game state.
        Args:
            state: The current game state.
            all_rewards: The rewards received after taking the action.
        Returns:
            VideoCheckersInfo: Additional information about the game state.
        """
        return VideoCheckersInfo(all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: VideoCheckersState, state: VideoCheckersState):
        """
        Returns the environment reward based on the game state.
        Args:
            previous_state: The previous game state.
        """
        return 0 # TODO: Implement environment reward logic

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: VideoCheckersState, state: VideoCheckersState):
        """
        Returns all rewards based on the game state.
        Args:
            previous_state: The previous game state.
            state: The current game state.
        Returns:
            rewards: The rewards received after taking the action.
        """
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: VideoCheckersState) -> bool:
        """
        Returns whether the game is done based on the game state.
        Args:
            state: The current game state.
        """
        return False  # TODO: Implement game over logic


def load_sprites():
    """
    Load all sprites required for Flag Capture rendering.
    Returns:
        TODO
    """
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    background = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocheckers/background.npy"), transpose=True)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(background, axis=0)
    SPRITE_PIECES = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/videocheckers/pieces/{}.npy"),
                                           num_chars=7)
    SPRITE_TEXT = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/videocheckers/text/{}.npy"),
                                          num_chars=12)

    return (
        SPRITE_BG,
        SPRITE_PIECES,
        SPRITE_TEXT,
    )


class VideoCheckersRenderer(AtraJaxisRenderer):
    def __init__(self):
        super().__init__()
        (
            self.SPRITE_BG,
            self.SPRITE_PIECES,
            self.SPRITE_TEXT,
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A VideoCheckersState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        raster: jnp.ndarray = jnp.full((WIDTH, HEIGHT, 3), jnp.array([160, 96, 64], dtype=jnp.uint8))

        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, OFFSET_X_BOARD, OFFSET_Y_BOARD, frame_bg)

        # Render the pieces on the board
        # position 0 is bottom right, position 31 is top left. right to left, bottom to top
        for i in range(32):
            piece_type = state.board[i]
            piece_frame = aj.get_sprite_frame(self.SPRITE_PIECES, piece_type)
            if piece_frame is not None:
                # Calculate the position on the board. 1 is at 1G, 2 is at 1E, 3 is at 1C, # 4 is at 1A, 5 is at 2H, 6 is at 2F, etc.
                row = i // 4
                col = (i % 4) * 2 + (1 if row % 2 == 0 else 0)
                x = OFFSET_X_BOARD + 4 + col * 17
                y = OFFSET_Y_BOARD + 2 + row * 13
                raster = aj.render_at(raster, x, y, piece_frame)

        return raster


def get_human_action() -> chex.Array:
    """
    Records human input for the game.

    Returns:
        action: int, action taken by the player
    """
    keys = pygame.key.get_pressed()
    # pygame.K_a  Links
    # pygame.K_d  Rechts
    # pygame.K_w  Hoch
    # pygame.K_s  Runter
    # pygame.K_SPACE  Aufdecken (fire)

    pressed_buttons = 0
    if keys[pygame.K_a]:
        pressed_buttons += 1
    if keys[pygame.K_d]:
        pressed_buttons += 1
    if keys[pygame.K_w]:
        pressed_buttons += 1
    if keys[pygame.K_s]:
        pressed_buttons += 1
    if pressed_buttons > 3:
        print("You have pressed a physically impossible combination of buttons")
        return jnp.array(Action.NOOP)

    if keys[pygame.K_a]:
        return jnp.array(Action.UPLEFT)
    if keys[pygame.K_d]:
        return jnp.array(Action.UPRIGHT)
    if keys[pygame.K_w]:
        return jnp.array(Action.DOWNLEFT)
    if keys[pygame.K_s]:
        return jnp.array(Action.DOWNRIGHT)
    if keys[pygame.K_SPACE]:
        return jnp.array(Action.FIRE)
    return jnp.array(Action.NOOP)



if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flag Capture Game")
    clock = pygame.time.Clock()

    game = JaxVideoCheckers()

    # Create the JAX renderer
    renderer = VideoCheckersRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                    event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(30)

    pygame.quit()