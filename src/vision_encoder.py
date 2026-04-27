import numpy as np


class VisionEncoder:
    """
    Converts raw drone state into natural language description.
    
    In a real VLA system this would process actual camera frames.
    Here we translate the simulation state into text the LLM can reason about.
    
    State: [drone_x, drone_y, vel_x, vel_y, platform_x, platform_vel]
    All values are normalized — we denormalize for human-readable description.
    """

    # Denormalization constants (matching drone_env.py)
    X_SCALE   = 10.0
    Y_SCALE   = 15.0
    VX_SCALE  = 10.0
    VY_SCALE  = 10.0
    PX_SCALE  = 10.0
    PV_SCALE  = 5.0

    def encode(self, state: np.ndarray) -> str:
        """
        Convert normalized state vector to natural language scene description.
        """
        x, y, vx, vy, px, pv = state

        # Denormalize
        x  *= self.X_SCALE
        y  *= self.Y_SCALE
        vx *= self.VX_SCALE
        vy *= self.VY_SCALE
        px *= self.PX_SCALE
        pv *= self.PV_SCALE

        # Spatial relationships
        dx          = x - px
        altitude    = y - 1.0  # height above platform
        approaching = vy < 0   # moving downward

        # Build description
        description = self._build_description(
            x, y, vx, vy, px, pv, dx, altitude, approaching
        )
        return description

    def _build_description(self, x, y, vx, vy, px, pv,
                           dx, altitude, approaching):

        # Horizontal position relative to platform
        if abs(dx) < 0.5:
            h_pos = "directly above the platform"
        elif dx > 0:
            h_pos = f"{abs(dx):.1f}m to the RIGHT of the platform"
        else:
            h_pos = f"{abs(dx):.1f}m to the LEFT of the platform"

        # Altitude description
        if altitude < 0.5:
            alt_desc = "very close to the platform (landing zone)"
        elif altitude < 3.0:
            alt_desc = f"{altitude:.1f}m above the platform"
        else:
            alt_desc = f"high up at {altitude:.1f}m above the platform"

        # Vertical motion
        if abs(vy) < 0.5:
            v_desc = "hovering vertically"
        elif vy < 0:
            v_desc = f"descending at {abs(vy):.1f} m/s"
        else:
            v_desc = f"rising at {abs(vy):.1f} m/s"

        # Horizontal motion
        if abs(vx) < 0.5:
            h_desc = "stable horizontally"
        elif vx > 0:
            h_desc = f"drifting RIGHT at {abs(vx):.1f} m/s"
        else:
            h_desc = f"drifting LEFT at {abs(vx):.1f} m/s"

        # Platform motion
        if abs(pv) < 0.5:
            p_desc = "stationary"
        elif pv > 0:
            p_desc = f"moving RIGHT at {abs(pv):.1f} m/s"
        else:
            p_desc = f"moving LEFT at {abs(pv):.1f} m/s"

        return (
            f"DRONE STATUS:\n"
            f"- Position: {h_pos}, {alt_desc}\n"
            f"- Motion: {v_desc}, {h_desc}\n"
            f"- Platform: {p_desc}\n"
            f"- Drone coords: ({x:.1f}, {y:.1f}) | "
            f"Platform: {px:.1f} | "
            f"Velocity: ({vx:.1f}, {vy:.1f})"
        )


if __name__ == '__main__':
    encoder = VisionEncoder()

    # Test with a sample state
    test_state = np.array([0.3, 0.6, -0.1, -0.2, 0.1, 0.3])
    description = encoder.encode(test_state)
    print("=== Vision Encoder Test ===\n")
    print(description)