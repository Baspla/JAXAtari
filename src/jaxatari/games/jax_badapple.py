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

NUM_FRAMES = 6572


class VideoState(NamedTuple):
    frame: chex.Array


class VideoObservation(NamedTuple):
    frame: chex.Array


class VideoInfo(NamedTuple):
    all_rewards: chex.Array


class JaxVideo(JaxEnvironment[VideoState, VideoObservation, VideoInfo]):
    def __init__(self, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            JAXAtariAction.NOOP
        }

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(187)) -> Tuple[
        VideoObservation, VideoState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)

        Args:
            key: Random key for generating the initial state.
        Returns:
            initial_obs: Initial observation of the game.
            state: Initial game state.
        """
        state = VideoState(
            frame=jnp.array(0).astype(jnp.int32),
        )
        initial_obs = self._get_observation(state)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: VideoState):
        """
        Returns the observation of the game state.
        Args:
            state: The current game state.
        Returns:
            VideoObservation: The observation of the game state.
        """
        return VideoObservation(
            frame=state.frame
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: VideoState, action: chex.Array) -> Tuple[
        VideoObservation, VideoState, float, bool, VideoInfo]:
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
        new_frame = state.frame + 1
        new_state = VideoState(
            frame=new_frame,
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        """
        Returns the action space of the game environment.
        Returns:
            action_space: The action space of the game environment.
        """
        return jnp.array(list(self.action_set), dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: VideoState, all_rewards: chex.Array) -> VideoInfo:
        """
        Returns additional information about the game state.
        Args:
            state: The current game state.
            all_rewards: The rewards received after taking the action.
        Returns:
            VideoInfo: Additional information about the game state.
        """
        return VideoInfo(all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: VideoState, state: VideoState):
        """
        Returns the environment reward based on the game state.
        Args:
            previous_state: The previous game state.
        """
        return state.frame - previous_state.frame

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: VideoState, state: VideoState):
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
    def _get_done(self, state: VideoState) -> bool:
        """
        Returns whether the game is done based on the game state.
        Args:
            state: The current game state.
        """
        return state.frame >= 6572


def load_sprites():
    """
    Load all sprites required for Video rendering.
    Returns:
        SPRITE_VIDEO
    """
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    SPRITE_VIDEO = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/badapple/{}.npy"),
                                          num_chars=6572)
    jax.debug.print("SPRITE_VIDEO: {SPRITE_VIDEO}", SPRITE_VIDEO=SPRITE_VIDEO.shape)
    return SPRITE_VIDEO



class VideoRenderer(AtraJaxisRenderer):
    def __init__(self):
        self.SPRITE_VIDEO = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A VideoState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # Create empty raster with CORRECT orientation for atraJaxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen
        raster: jnp.ndarray = jnp.zeros((WIDTH, HEIGHT, 3))
        raster = aj.render_at(raster, 0, 0, self.SPRITE_VIDEO[state.frame])
        return raster


def get_human_action() -> chex.Array:
    """
    Records human input for the game.

    Returns:
        action: int, action taken by the player (FIRE)
    """
    keys = pygame.key.get_pressed()
    return jnp.array(JAXAtariAction.NOOP)


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Bad Apple")
    clock = pygame.time.Clock()

    game = JaxVideo()

    # Create the JAX renderer
    renderer = VideoRenderer()

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
