import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame

from jaxatari.environment import JAXAtariAction

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


class VideoCheckersState(NamedTuple):
    cursor_pos: chex.Array
    board: chex.Array # Shape (32,1) for 8x8 board only black fields and 2 dimension for piece type.
    selected_piece: chex.Array
    animation_frame: chex.Array

class VideoCheckersObservation(NamedTuple):
    board: chex.Array # All animation is already baked into the board observation
    start_pos: chex.Array
    end_pos: chex.Array

class VideoCheckersInfo(NamedTuple):
    all_rewards: chex.Array


class JaxVideoCheckers(JaxEnvironment[VideoCheckersState, VideoCheckersObservation, VideoCheckersInfo]):
    def __init__(self, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            JAXAtariAction.FIRE,
            JAXAtariAction.UPRIGHT,
            JAXAtariAction.UPLEFT,
            JAXAtariAction.DOWNRIGHT,
            JAXAtariAction.DOWNLEFT
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
        return jnp.array(JAXAtariAction.NOOP)

    if keys[pygame.K_a]:
        return jnp.array(JAXAtariAction.UPLEFT)
    if keys[pygame.K_d]:
        return jnp.array(JAXAtariAction.UPRIGHT)
    if keys[pygame.K_w]:
        return jnp.array(JAXAtariAction.DOWNLEFT)
    if keys[pygame.K_s]:
        return jnp.array(JAXAtariAction.DOWNRIGHT)
    if keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.FIRE)
    return jnp.array(JAXAtariAction.NOOP)



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
