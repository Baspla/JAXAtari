from functools import partial
import os
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


class WizardOfWorConstants(NamedTuple):
    WINDOW_WIDTH: int = 160
    WINDOW_HEIGHT: int = 210
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER_COLOR: Tuple[int, int, int] = (255, 255, 0)
    ENEMY_COLOR: Tuple[int, int, int] = (255, 0, 0)
    BULLET_COLOR: Tuple[int, int, int] = (255, 255, 255)
    WALL_COLOR: Tuple[int, int, int] = (0, 0, 255)
    # Sprite sizes (Platzhalter)
    PLAYER_SIZE: Tuple[int, int] = (8, 8)
    ENEMY_SIZE: Tuple[int, int] = (8, 8)
    BULLET_SIZE: Tuple[int, int] = (2, 2)
    WALL_THICKNESS: int = 2
    GAMEBOARD_1_WALLS_HORIZONTAL = jnp.array([
        [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]])
    GAMEBOARD_1_WALLS_VERTICAL = jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]])
    # Richtungen
    UP: int = Action.UP
    DOWN: int = Action.DOWN
    LEFT: int = Action.LEFT
    RIGHT: int = Action.RIGHT

    # IMPORTANT: About the coordinates
    # The board goes from 0,0 (top-left) to 60,110 (bottom-right)
    @staticmethod
    def get_walls_for_gameboard(gameboard: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """    Placeholder: Gibt die Wände für das angegebene Gameboard zurück.
        :param gameboard: Das Gameboard, für das die Wände abgerufen werden sollen.
        :return: Ein Tupel mit den horizontalen und vertikalen Wänden.
        """
        return jax.lax.cond(
            gameboard == 1,
            lambda _: (WizardOfWorConstants.GAMEBOARD_1_WALLS_HORIZONTAL, WizardOfWorConstants.GAMEBOARD_1_WALLS_VERTICAL),
            lambda _: (jnp.zeros((5, 11), dtype=jnp.int32), jnp.zeros((6, 10), dtype=jnp.int32)),
            operand=None
        )# Hier können weitere Gameboards hinzugefügt werden


class EntityPosition(NamedTuple):
    x: chex.Array
    y: chex.Array
    direction: int  # Richtung aus UP, DOWN, LEFT, RIGHT


class WizardOfWorObservation(NamedTuple):
    player: EntityPosition
    enemies: chex.Array
    bullet: EntityPosition
    score: chex.Array
    lives: chex.Array


class WizardOfWorInfo(NamedTuple):
    all_rewards: chex.Array


class WizardOfWorState(NamedTuple):
    # Spielfigur
    player: EntityPosition
    # Gegner, Schüsse etc.
    enemies: chex.Array
    gameboard: int
    bullet: EntityPosition
    score: chex.Array
    lives: int


def updateState(state: WizardOfWorState, player: EntityPosition = None, enemies: chex.Array = None,
                gameboard: int = None, bullet: EntityPosition = None, score: chex.Array = None,
                lives: int = None) -> WizardOfWorState:
    """
    Aktualisiert den Zustand des Spiels. Nur diese Methode sollte verwendet werden, um das State Objekt zu mutieren.
    Nicht übergebene Parameter werden aus dem aktuellen Zustand übernommen.
    :param state: Der aktuelle Zustand des Spiels.
    :param player: Neue Position der Spielfigur.
    :param enemies: Neue Positionen der Gegner.
    :param gameboard: Neues Gameboard.
    :param bullet: Neue Position des Schusses.
    :param score: Neuer Punktestand.
    :param lives: Neue Anzahl der Leben.
    :return: Ein neuer Zustand des Spiels mit den aktualisierten Werten.
    """
    return WizardOfWorState(
        player=player if player is not None else state.player,
        enemies=enemies if enemies is not None else state.enemies,
        gameboard=gameboard if gameboard is not None else state.gameboard,
        bullet=bullet if bullet is not None else state.bullet,
        score=score if score is not None else state.score,
        lives=lives if lives is not None else state.lives
    )


class JaxWizardOfWor(JaxEnvironment[WizardOfWorState, WizardOfWorObservation, WizardOfWorInfo, WizardOfWorConstants]):
    def __init__(self, consts: WizardOfWorConstants = None, reward_funcs: list[callable] = None):
        consts = consts or WizardOfWorConstants()
        super().__init__(consts)
        self.renderer = WizardOfWorRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[WizardOfWorObservation, WizardOfWorState]:
        state = WizardOfWorState(
            player=EntityPosition(
                x=jnp.array(self.consts.WINDOW_WIDTH // 2),
                y=jnp.array(self.consts.WINDOW_HEIGHT // 2),
                direction=self.consts.LEFT
            ),
            enemies=jnp.zeros((0, 4), dtype=jnp.int32),  # Platzhalter für Gegner
            gameboard=1,  # Start mit Gameboard 1
            bullet=EntityPosition(
                x=jnp.array(-1),  # Keine Schüsse zu Beginn
                y=jnp.array(-1),
                direction=self.consts.UP
            ),
            score=jnp.array(0),
            lives=3  # Startleben
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: WizardOfWorState, action: chex.Array) -> Tuple[
        WizardOfWorObservation, WizardOfWorState, chex.Array, chex.Array, WizardOfWorInfo]:
        new_state = previous_state = state
        new_state = self._step_level_change(new_state)
        new_state = self._step_respawn(new_state, action)
        new_state = self._step_player_movement(new_state, action)
        new_state = self._step_bullet_movement(new_state)
        new_state = self._step_enemy_movement(new_state)
        new_state = self._step_collision_detection(new_state)
        done = self._get_done(new_state)
        env_reward = self._get_reward(previous_state, new_state)
        all_rewards = self._get_all_reward(previous_state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def render(self, state: WizardOfWorState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)  # Platzhalter

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WINDOW_WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.WINDOW_HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WINDOW_WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.WINDOW_HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "enemies": spaces.Box(low=0, high=self.consts.WINDOW_WIDTH, shape=(None, 4), dtype=jnp.int32),
            "bullets": spaces.Box(low=0, high=self.consts.WINDOW_WIDTH, shape=(None, 4), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=10, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: WizardOfWorState) -> chex.Array:
        return jnp.array(False)  # später implementieren

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        return jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: WizardOfWorState, all_rewards: chex.Array = None) -> WizardOfWorInfo:
        return WizardOfWorInfo(all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState) -> chex.Array:
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: WizardOfWorState) -> WizardOfWorObservation:
        player_entity = EntityPosition(
            x=state.player.x,
            y=state.player.y,
            direction=state.player.direction
        )
        enemies = state.enemies  # Placeholder for enemy positions
        bullet_entity = EntityPosition(
            x=state.bullet.x,
            y=state.bullet.y,
            direction=state.bullet.direction
        )
        return WizardOfWorObservation(
            player=player_entity,
            enemies=enemies,
            bullet=bullet_entity,
            score=state.score,
            lives=jnp.array(state.lives)
        )

    def _step_level_change(self, state):
        # Placeholder: Logik für Levelwechsel.
        # Unser MVP hat nur Level 1, daher wird hier nichts geändert
        return state

    def _step_respawn(self, state, action):
        # Placeholder: Logik für Respawn
        # hier wird geprüft, ob der tot ist und durch einen Input respawned werden soll

        return state

    def _step_player_movement(self, state, action):
        # Placeholder: Spielerbewegung basierend auf der Aktion
        # also Bewegung, Rotation und Kollision mit Wänden
        return state

    def _step_bullet_movement(self, state):
        # Placeholder: Schussbewegung
        # hier wird die Position des Schusses aktualisiert
        return state

    def _step_enemy_movement(self, state):
        return state

    def _step_collision_detection(self, state):
        return state

class WizardOfWorRenderer(JAXGameRenderer):
    def __init__(self, consts: WizardOfWorConstants = None):
        super().__init__()
        self.consts = consts or WizardOfWorConstants()
        # Placeholder: Sprites laden
        self.SPRITE_BG = jnp.zeros((1, self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: WizardOfWorState):
        # Raster initialisieren
        raster = jr.create_initial_frame(width=self.consts.WINDOW_WIDTH, height=self.consts.WINDOW_HEIGHT)
        raster = self._render_gameboard(raster, state)
        raster = self._render_enemies(raster, state)
        raster = self._render_bullet(raster, state)
        raster = self._render_player(raster, state)
        raster = self._render_score(raster, state)
        raster = self._render_lives(raster, state)
        return raster

    def _render_gameboard(self, raster, state:WizardOfWorState):
        # Placeholder: Hintergrund und Wände zeichnen

        def _render_gameboard_background(raster):
            # Hintergrund zeichnen
            return raster

        def _render_gameboard_walls(raster, gamestate: WizardOfWorState):
            # Wände zeichnen basierend auf dem Gameboard
            walls_horizontal, walls_vertical = self.consts.get_walls_for_gameboard(gameboard=gamestate.gameboard)
            return raster

        _render_gameboard_background(raster)
        _render_gameboard_walls(raster, state)
        return raster

    def _render_enemies(self, raster, state):
        # Placeholder: Gegner zeichnen
        return raster

    def _render_bullet(self, raster, state):
        # Placeholder: Schuss zeichnen
        return raster

    def _render_player(self, raster, state):
        # Placeholder: Spieler zeichnen
        return raster

    def _render_score(self, raster, state):
        # Placeholder: Punktestand zeichnen
        return raster

    def _render_lives(self, raster, state):
        # Placeholder: Leben zeichnen
        return raster
