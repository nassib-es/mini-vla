import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

class DroneEnv:
    """
    2D Drone Landing Environment
    
    State:  [drone_x, drone_y, vel_x, vel_y, platform_x, platform_vel_x]
    Action: 0=thrust_left, 1=thrust_up, 2=thrust_right, 3=no_thrust
    Goal:   Land on the moving platform gently
    """

    # Environment constants
    GRAVITY        = -9.8
    THRUST         = 15.0
    DRAG           = 0.99
    DT             = 0.05       # time step seconds
    MAX_STEPS      = 500
    PLATFORM_WIDTH = 1.5
    DRONE_RADIUS   = 0.3

    # World bounds
    X_MIN, X_MAX = -10.0, 10.0
    Y_MIN, Y_MAX =   0.0, 15.0
    PLATFORM_Y   =   1.0        # platform height

    # Reward constants
    REWARD_LAND         =  100.0
    REWARD_CRASH        = -100.0
    REWARD_OUT_OF_BOUNDS= -50.0
    REWARD_STEP         =  -0.1  # small penalty per step to encourage speed

    def __init__(self, platform_speed=1.5, random_platform=True):
        self.platform_speed  = platform_speed
        self.random_platform = random_platform
        self.state           = None
        self.steps           = 0
        self.reset()

    def normalize_state(self, state):
        state[0] /= 10.0
        state[1] /= 15.0
        state[2] /= 10.0
        state[3] /= 10.0
        state[4] /= 10.0
        state[5] /= 5.0
        return state

    def reset(self):
        # Drone starts at random position near top
        drone_x   = np.random.uniform(-5, 5)
        drone_y   = np.random.uniform(8, 13)
        vel_x     = np.random.uniform(-2, 2)
        vel_y     = np.random.uniform(-2, 0)

        # Platform starts at random position
        platform_x   = np.random.uniform(-4, 4) if self.random_platform else 0.0
        platform_vel = self.platform_speed * np.random.choice([-1, 1])

        self.state = np.array([
            drone_x, drone_y, vel_x, vel_y,
            platform_x, platform_vel
        ], dtype=np.float32)

        self.steps = 0
        return self.normalize_state(self.state.copy())

    def step(self, action):
        x, y, vx, vy, px, pv = self.state

        # Apply thrust based on action
        ax, ay = 0.0, 0.0
        if action == 0:   ax = -self.THRUST   # left
        elif action == 1: ay =  self.THRUST   # up
        elif action == 2: ax =  self.THRUST   # right
        # action == 3: no thrust

        # Physics update
        ay += self.GRAVITY
        vx  = (vx + ax * self.DT) * self.DRAG
        vy  = (vy + ay * self.DT) * self.DRAG
        x  +=  vx * self.DT
        y  +=  vy * self.DT

        # Platform movement — bounces off walls
        px += pv * self.DT
        if px <= self.X_MIN + 1 or px >= self.X_MAX - 1:
            pv = -pv

        self.steps += 1
        self.state = np.array([x, y, vx, vy, px, pv], dtype=np.float32)

        # Check termination
        done, reward = self._check_termination(x, y, vx, vy, px)

        return self.normalize_state(self.state.copy()), reward, done

    def _check_termination(self, x, y, vx, vy, px):
        # Out of bounds
        if x < self.X_MIN or x > self.X_MAX or y > self.Y_MAX:
            return True, self.REWARD_OUT_OF_BOUNDS

        # Crashed into ground
        if y <= 0:
            return True, self.REWARD_CRASH

        # Landed on platform
        on_platform = abs(x - px) < self.PLATFORM_WIDTH / 2
        near_platform = abs(y - self.PLATFORM_Y) < 0.3
        gentle = abs(vy) < 3.0 and abs(vx) < 3.0

        if on_platform and near_platform:
            if gentle:
                return True, self.REWARD_LAND
            else:
                return True, self.REWARD_CRASH  # too fast = crash

        # Max steps
        if self.steps >= self.MAX_STEPS:
            return True, self.REWARD_STEP * 10

        return False, self.REWARD_STEP

    @property
    def state_size(self):
        return 6

    @property
    def action_size(self):
        return 4