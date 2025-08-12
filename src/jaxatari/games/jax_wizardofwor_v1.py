# ---------------------------------------------------------------------
# jax_wizard_of_wor.py
# ---------------------------------------------------------------------
# Single-player Wizard-of-Wor (Level 1) for the JAXAtari framework
#
# Code style mirrors jax_pong.py:
# - NamedTuple data containers
# - self.consts everywhere (including pure game logic)
# - Pure game logic as bound methods of the Env
# - jax_rendering_utils (jr) for sprite-based rendering
# - JIT-friendly code: jax.jit, jax.lax.cond/select, jnp ops, no Python int()
# - Named parameters in function calls for readability
# ---------------------------------------------------------------------

import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


# =====================================================================
# 1. CONSTANTS
# =====================================================================

class WizardOfWorConstants(NamedTuple):
    # Maze geometry (grid coordinates)
    MAZE_W: int = 11
    MAZE_H: int = 6
    WALL_TEMPLATE: chex.Array = jnp.asarray(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=jnp.bool_,
    )

    # Rendering
    WIDTH: int = 160
    HEIGHT: int = 210
    TILE_SIZE: int = 8
    GRID_OFFSET_X: int = 20
    GRID_OFFSET_Y: int = 30

    # Gameplay
    MAX_BULLETS: int = 4
    N_MONSTERS: int = 6
    PLAYER_LIVES: int = 3
    MONSTER_PTS: int = 100
    PLAYER_START: chex.Array = jnp.asarray([MAZE_W // 2, MAZE_H - 2], dtype=jnp.int32)

    # Fallback sprite colors
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    WALL_COLOR: Tuple[int, int, int] = (40, 40, 40)
    PLAYER_COLOR: Tuple[int, int, int] = (220, 220, 40)
    MONSTER_COLOR: Tuple[int, int, int] = (30, 160, 255)
    BULLET_COLOR: Tuple[int, int, int] = (240, 40, 40)
    SCORE_COLOR: Tuple[int, int, int] = (255, 255, 255)

    # Action mapping (uses JAXAtariAction)
    NOOP: int = Action.NOOP
    FIRE: int = Action.FIRE
    UP: int = Action.UP
    RIGHT: int = Action.RIGHT
    LEFT: int = Action.LEFT
    DOWN: int = Action.DOWN
    UPFIRE: int = Action.UPFIRE
    RIGHTFIRE: int = Action.RIGHTFIRE
    LEFTFIRE: int = Action.LEFTFIRE
    DOWNFIRE: int = Action.DOWNFIRE

    # Lookup tables (grid deltas)
    MOVE_DELTAS: chex.Array = jnp.asarray(
        [
            [0, 0],   # NOOP
            [0, 0],   # FIRE
            [0, -1],  # UP
            [1, 0],   # RIGHT
            [-1, 0],  # LEFT
            [0, 1],   # DOWN
            [0, -1],  # UPFIRE
            [1, 0],   # RIGHTFIRE
            [-1, 0],  # LEFTFIRE
            [0, 1],   # DOWNFIRE
        ],
        dtype=jnp.int32,
    )

    DIR_FROM_ACT: chex.Array = jnp.asarray(
        [0, 0, 0, 1, 3, 2, 0, 1, 3, 2], dtype=jnp.int32
    )


# =====================================================================
# 2. DATA CONTAINERS
# =====================================================================

class WizardOfWorState(NamedTuple):
    # Player
    player_pos: chex.Array                # (2,) int32 (x, y) in grid units
    player_dir: chex.Array                # int32: 0↑ 1→ 2↓ 3←
    player_lives: chex.Array              # int32
    player_score: chex.Array              # int32
    # Monsters
    monsters_pos: chex.Array              # (N_MONSTERS,2) int32
    monsters_alive: chex.Array            # (N_MONSTERS,) bool
    monsters_dir: chex.Array              # (N_MONSTERS,) int32
    # Bullets: x,y,dir,active
    bullets: chex.Array                   # (MAX_BULLETS,4) int32
    # World
    walls: chex.Array                     # (MAZE_W,MAZE_H) bool
    rng: chex.Array                       # PRNGKey (int32[2])
    timestep: chex.Array                  # int32
    done: chex.Array                      # bool


class WizardOfWorObservation(NamedTuple):
    frame: chex.Array                     # (H,W,3) uint8


class WizardOfWorInfo(NamedTuple):
    score: chex.Array
    lives: chex.Array
    level_complete: chex.Array


# =====================================================================
# 3. RENDERER
# =====================================================================

class WizardOfWorRenderer(JAXGameRenderer):
    def __init__(self, consts: WizardOfWorConstants = None):
        super().__init__()
        self.consts = consts or WizardOfWorConstants()
        (
            self.SPRITE_PLAYER,
            self.SPRITE_MONSTER,
            self.SPRITE_BULLET,
            self.SPRITE_WALL,
            self.DIGIT_SPRITES,
        ) = self.load_sprites()

    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        try:
            player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizard_of_wor/player.npy"))
            monster = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizard_of_wor/monster.npy"))
            bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizard_of_wor/bullet.npy"))
            wall = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizard_of_wor/wall.npy"))

            SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
            SPRITE_MONSTER = jnp.expand_dims(monster, axis=0)
            SPRITE_BULLET = jnp.expand_dims(bullet, axis=0)
            SPRITE_WALL = jnp.expand_dims(wall, axis=0)

            DIGIT_SPRITES = jr.load_and_pad_digits(
                os.path.join(MODULE_DIR, "sprites/wizard_of_wor/digit_{}.npy"),
                num_chars=10,
            )
        except Exception:
            # Fallback solid-color sprites (HWC RGBA)
            SPRITE_PLAYER = jnp.expand_dims(self._create_simple_sprite(color=self.consts.PLAYER_COLOR, size=self.consts.TILE_SIZE), axis=0)
            SPRITE_MONSTER = jnp.expand_dims(self._create_simple_sprite(color=self.consts.MONSTER_COLOR, size=self.consts.TILE_SIZE), axis=0)
            SPRITE_BULLET = jnp.expand_dims(self._create_simple_sprite(color=self.consts.BULLET_COLOR, size=max(2, self.consts.TILE_SIZE // 2)), axis=0)
            SPRITE_WALL = jnp.expand_dims(self._create_simple_sprite(color=self.consts.WALL_COLOR, size=self.consts.TILE_SIZE), axis=0)
            DIGIT_SPRITES = self._create_fallback_digits()

        return (
            SPRITE_PLAYER,
            SPRITE_MONSTER,
            SPRITE_BULLET,
            SPRITE_WALL,
            DIGIT_SPRITES,
        )

    def _create_simple_sprite(self, color, size=8):
        rgba = jnp.zeros((size, size, 4), dtype=jnp.uint8)
        rgba = rgba.at[:, :, :3].set(jnp.array(color, dtype=jnp.uint8))
        rgba = rgba.at[:, :, 3].set(255)
        return rgba

    def _create_fallback_digits(self):
        digits = []
        for _ in range(10):
            digit = jnp.zeros((8, 8, 4), dtype=jnp.uint8)
            digit = digit.at[:, :, :3].set(jnp.array(self.consts.SCORE_COLOR, dtype=jnp.uint8))
            digit = digit.at[:, :, 3].set(255)
            digits.append(digit)
        return jnp.array(digits)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=self.consts.WIDTH, height=self.consts.HEIGHT)

        # Walls from mask
        def render_wall_tile(i, current_raster):
            wall_x = i % self.consts.MAZE_W
            wall_y = i // self.consts.MAZE_W
            is_wall = state.walls[wall_x, wall_y]

            screen_x = wall_x * self.consts.TILE_SIZE + self.consts.GRID_OFFSET_X
            screen_y = wall_y * self.consts.TILE_SIZE + self.consts.GRID_OFFSET_Y

            frame_wall = jr.get_sprite_frame(self.SPRITE_WALL, 0)
            return jax.lax.cond(
                is_wall,
                lambda f: jr.render_at(f, screen_x, screen_y, frame_wall),
                lambda f: f,
                current_raster,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAZE_W * self.consts.MAZE_H, render_wall_tile, raster)

        # Monsters
        def render_monster(i, current_raster):
            is_alive = state.monsters_alive[i]
            pos = state.monsters_pos[i]
            screen_x = pos[0] * self.consts.TILE_SIZE + self.consts.GRID_OFFSET_X
            screen_y = pos[1] * self.consts.TILE_SIZE + self.consts.GRID_OFFSET_Y

            frame_monster = jr.get_sprite_frame(self.SPRITE_MONSTER, 0)
            return jax.lax.cond(
                is_alive,
                lambda f: jr.render_at(f, screen_x, screen_y, frame_monster),
                lambda f: f,
                current_raster,
            )

        raster = jax.lax.fori_loop(0, self.consts.N_MONSTERS, render_monster, raster)

        # Player
        player_screen_x = state.player_pos[0] * self.consts.TILE_SIZE + self.consts.GRID_OFFSET_X
        player_screen_y = state.player_pos[1] * self.consts.TILE_SIZE + self.consts.GRID_OFFSET_Y
        frame_player = jr.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = jr.render_at(raster, player_screen_x, player_screen_y, frame_player)

        # Bullets
        def render_bullet(i, current_raster):
            is_active = state.bullets[i, 3] == 1
            pos = state.bullets[i, :2]
            screen_x = pos[0] * self.consts.TILE_SIZE + self.consts.GRID_OFFSET_X
            screen_y = pos[1] * self.consts.TILE_SIZE + self.consts.GRID_OFFSET_Y

            frame_bullet = jr.get_sprite_frame(self.SPRITE_BULLET, 0)
            return jax.lax.cond(
                is_active,
                lambda f: jr.render_at(f, screen_x, screen_y, frame_bullet),
                lambda f: f,
                current_raster,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_BULLETS, render_bullet, raster)

        # HUD: score
        score_digits = jr.int_to_digits(state.player_score, max_digits=6)
        raster = jr.render_label(raster, 10, 10, score_digits, self.DIGIT_SPRITES, spacing=8)

        # HUD: lives
        raster = jr.render_indicator(
            raster,
            10,
            self.consts.HEIGHT - 20,
            state.player_lives,
            jr.get_sprite_frame(self.SPRITE_PLAYER, 0),
            spacing=12,
            )

        return raster


# =====================================================================
# 4. ENVIRONMENT WRAPPER (pure logic as bound methods using self.consts)
# =====================================================================

class JaxWizardOfWor(JaxEnvironment[WizardOfWorState, WizardOfWorObservation, WizardOfWorInfo, WizardOfWorConstants]):
    def __init__(self, consts: WizardOfWorConstants = None, reward_funcs: list[callable] = None):
        consts = consts or WizardOfWorConstants()
        super().__init__(consts)
        self.renderer = WizardOfWorRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

    # ------------------------- pure game logic -------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _wall_at(self, walls, pos):
        inside = (
                (pos[0] >= 0) & (pos[0] < self.consts.MAZE_W) &
                (pos[1] >= 0) & (pos[1] < self.consts.MAZE_H)
        )
        return jax.lax.cond(
            inside,
            lambda _: walls[pos[0], pos[1]],
            lambda _: True,
            operand=None,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _move_entity(self, pos, delta, walls):
        new_pos = pos + delta
        blocked = self._wall_at(walls=walls, pos=new_pos)
        return jax.lax.cond(
            blocked,
            lambda _: pos,
            lambda _: new_pos,
            operand=None,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_bullet(self, state, direction):
        def fill_slot(i, b):
            is_free = b[i, 3] == 0
            datum = jnp.asarray([state.player_pos[0], state.player_pos[1], direction, 1], dtype=jnp.int32)
            return jax.lax.cond(
                is_free,
                lambda _: b.at[i].set(datum),
                lambda _: b,
                operand=None,
            )

        return jax.lax.fori_loop(0, self.consts.MAX_BULLETS, fill_slot, state.bullets)

    @partial(jax.jit, static_argnums=(0,))
    def _update_bullets(self, bullets, walls):
        def step_single(b):
            active = b[3] == 1
            dx = jnp.asarray([0, 1, 0, -1], dtype=jnp.int32)[b[2]]
            dy = jnp.asarray([-1, 0, 1, 0], dtype=jnp.int32)[b[2]]
            new_pos = b[:2] + jnp.asarray([dx, dy], dtype=jnp.int32)
            deactivate = self._wall_at(walls=walls, pos=new_pos)
            keep = active & (~deactivate)
            return jax.lax.cond(
                keep,
                lambda _: jnp.asarray([new_pos[0], new_pos[1], b[2], 1], dtype=jnp.int32),
                lambda _: jnp.zeros(4, dtype=jnp.int32),
                operand=None,
            )

        return jax.vmap(step_single)(bullets)

    @partial(jax.jit, static_argnums=(0,))
    def _update_monsters(self, state):
        rng, sub_key = jax.random.split(state.rng)
        dirs = jax.random.randint(key=sub_key, shape=(self.consts.N_MONSTERS,), minval=0, maxval=4)
        dx = jnp.asarray([0, 1, 0, -1], dtype=jnp.int32)[dirs]
        dy = jnp.asarray([-1, 0, 1, 0], dtype=jnp.int32)[dirs]
        candidate = state.monsters_pos + jnp.stack([dx, dy], axis=1)
        blocked = jax.vmap(self._wall_at, in_axes=(None, 0))(state.walls, candidate)
        new_pos = jnp.where(blocked[:, None], state.monsters_pos, candidate)
        return new_pos, dirs, rng

    @partial(jax.jit, static_argnums=(0,))
    def _collision_logic(self, state):
        def bullet_hits(b):
            same = jnp.all(state.monsters_pos == b[:2], axis=1)
            return same & state.monsters_alive & (b[3] == 1)

        hit_matrix = jax.vmap(bullet_hits)(state.bullets)
        killed = jnp.any(hit_matrix, axis=0)
        new_alive = state.monsters_alive & (~killed)

        reward = (killed.sum() * self.consts.MONSTER_PTS).astype(jnp.int32)
        equal_xy = jnp.all(state.monsters_pos == state.player_pos[None, :], axis=1)
        touched = jnp.any(jnp.logical_and(equal_xy, new_alive))

        lives = jax.lax.cond(
            touched,
            lambda s: s - jnp.array(1, dtype=jnp.int32),
            lambda s: s,
            operand=state.player_lives,
        )
        dead = jnp.less_equal(lives, jnp.array(0, dtype=jnp.int32))
        return new_alive, reward, lives, dead

    # ------------------------- env lifecycle -------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _reset_state(self, key):
        key, x_key, y_key = jax.random.split(key, 3)
        mons_x = jax.random.randint(key=x_key, shape=(self.consts.N_MONSTERS,), minval=1, maxval=self.consts.MAZE_W - 1)
        mons_y = jax.random.randint(key=y_key, shape=(self.consts.N_MONSTERS,), minval=1, maxval=3)
        return WizardOfWorState(
            player_pos=self.consts.PLAYER_START.astype(jnp.int32),
            player_dir=jnp.array(0, dtype=jnp.int32),
            player_lives=jnp.array(self.consts.PLAYER_LIVES, dtype=jnp.int32),
            player_score=jnp.array(0, dtype=jnp.int32),
            monsters_pos=jnp.stack([mons_x, mons_y], axis=1).astype(jnp.int32),
            monsters_alive=jnp.ones((self.consts.N_MONSTERS,), dtype=jnp.bool_),
            monsters_dir=jnp.zeros((self.consts.N_MONSTERS,), dtype=jnp.int32),
            bullets=jnp.zeros((self.consts.MAX_BULLETS, 4), dtype=jnp.int32),
            walls=self.consts.WALL_TEMPLATE,
            rng=key,
            timestep=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
        )

    def reset(self, key=None) -> Tuple[WizardOfWorObservation, WizardOfWorState]:
        if key is None:
            key = jax.random.PRNGKey(0)
        state = self._reset_state(key=key)
        obs = self._get_observation(state=state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _step_core(self, state, action):
        # movement
        delta = self.consts.MOVE_DELTAS[action]
        new_player_pos = self._move_entity(pos=state.player_pos, delta=delta, walls=state.walls)
        player_dir = jax.lax.select(action == self.consts.NOOP, state.player_dir, self.consts.DIR_FROM_ACT[action])

        # shooting
        bullets = jax.lax.cond(
            (action == self.consts.FIRE) |
            (action == self.consts.UPFIRE) |
            (action == self.consts.RIGHTFIRE) |
            (action == self.consts.LEFTFIRE) |
            (action == self.consts.DOWNFIRE),
            lambda _: self._spawn_bullet(state=state, direction=player_dir),
            lambda _: state.bullets,
            operand=None,
            )
        bullets = self._update_bullets(bullets=bullets, walls=state.walls)

        # monsters
        mons_pos, mons_dir, rng = self._update_monsters(state=state._replace(rng=state.rng))

        # collisions
        new_alive, hit_reward, lives, dead = self._collision_logic(
            state=state._replace(
                player_pos=new_player_pos,
                bullets=bullets,
                monsters_pos=mons_pos,
                monsters_alive=state.monsters_alive,
            )
        )

        level_complete = jnp.all(~new_alive)
        done = jnp.logical_or(dead, level_complete)

        return state._replace(
            player_pos=new_player_pos,
            player_dir=player_dir,
            bullets=bullets,
            monsters_pos=mons_pos,
            monsters_dir=mons_dir,
            monsters_alive=new_alive,
            player_score=state.player_score + hit_reward,
            player_lives=lives,
            rng=rng,
            timestep=state.timestep + 1,
            done=done,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: WizardOfWorState, action: chex.Array) -> Tuple[WizardOfWorObservation, WizardOfWorState, float, bool, WizardOfWorInfo]:
        new_state = self._step_core(state=state, action=action)  # keep action as array
        observation = self._get_observation(state=new_state)
        env_reward = self._get_reward(previous_state=state, state=new_state)
        done = self._get_done(state=new_state)
        info = self._get_info(state=new_state)
        return observation, new_state, env_reward, done, info

    def render(self, state: WizardOfWorState) -> Tuple[jnp.ndarray]:
        return (self.renderer.render(state=state),)

    # ------------------------- spaces -------------------------

    def action_space(self) -> spaces.Discrete:
        # Single-player action set of size 10 (NOOP, FIRE, 4 dirs, 4 dir+FIRE)
        return spaces.Discrete(10)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def image_space(self) -> spaces.Box:
        return self.observation_space()

    # ------------------------- adapters: obs/info/reward/done -------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: WizardOfWorState) -> WizardOfWorObservation:
        return WizardOfWorObservation(frame=self.renderer.render(state=state))

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: WizardOfWorObservation) -> jnp.ndarray:
        return obs.frame.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: WizardOfWorState, all_rewards: chex.Array = None) -> WizardOfWorInfo:
        return WizardOfWorInfo(
            score=state.player_score,
            lives=state.player_lives,
            level_complete=jnp.all(~state.monsters_alive),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState):
        # dense reward: score delta
        return (state.player_score - previous_state.player_score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: WizardOfWorState) -> bool:
        return False
