import numpy as np
import sys
sys.path.append('.')

from src.vision_encoder import VisionEncoder
from src.llm_reasoner   import LLMReasoner
from src.action_decoder import ActionDecoder


class VLAAgent:
    """
    Hybrid Vision-Language-Action Agent.
    
    Architecture:
    - LLM reasons every N steps → sets high-level strategy
    - Rule-based controller executes low-level actions based on strategy
    
    This mirrors real robotics VLA systems where LLMs handle
    high-level planning and low-level controllers handle precise actuation.
    """

    STRATEGIES = {
        "ALIGN":   "Prioritize horizontal alignment with platform",
        "DESCEND": "Descend carefully toward platform",
        "BRAKE":   "Slow down — descent speed is dangerous",
        "LAND":    "Final approach — align and land gently",
    }

    def __init__(self, model="llama3.1", llm_every=10, verbose=True):
        self.encoder  = VisionEncoder()
        self.reasoner = LLMReasoner(model=model)
        self.decoder  = ActionDecoder()
        self.verbose  = verbose
        self.llm_every = llm_every

        self.step_count       = 0
        self.current_strategy = "ALIGN"

        if not self.reasoner.is_available():
            raise RuntimeError("Ollama not running! Start with: ollama serve")
        print(f"Hybrid VLA Agent | Model: {model} | LLM every {llm_every} steps")

    def act(self, state: np.ndarray) -> int:
        """
        Hybrid pipeline:
        - Every llm_every steps: LLM updates strategy
        - Every step: rule-based controller executes strategy
        """
        # Step 1 — LLM updates strategy periodically
        if self.step_count % self.llm_every == 0:
            scene    = self.encoder.encode(state)
            response = self.reasoner.decide(scene)
            strategy = self.reasoner.decode_strategy(response)
            if strategy:
                self.current_strategy = strategy
            if self.verbose:
                print(f"\n[LLM @ step {self.step_count}] "
                      f"Scene summary → Strategy: {self.current_strategy}")

        # Step 2 — Rule-based controller executes strategy
        action = self._rule_based_controller(state, self.current_strategy)

        if self.verbose:
            names = {0: "THRUST_LEFT", 1: "THRUST_UP",
                     2: "THRUST_RIGHT", 3: "NO_THRUST"}
            print(f"  Step {self.step_count:3d} | "
                  f"Strategy: {self.current_strategy:8s} | "
                  f"Action: {names[action]}")

        self.step_count += 1
        return action

    def _rule_based_controller(self, state, strategy) -> int:
        x, y, vx, vy, px, pv = state

        # Denormalize
        x  *= 10.0; y  *= 15.0
        vx *= 10.0; vy *= 10.0
        px *= 10.0

        dx       = x - px      # horizontal error
        altitude = y - 1.0     # height above platform
        speed    = abs(vy)     # descent speed

        # ── Emergency override ────────────────────────────────────────
        # If falling too fast anywhere — brake immediately
        if vy < -4.0:
            return 1  # THRUST_UP

        # ── LAND strategy — final approach ───────────────────────────
        if strategy == "LAND":
            # First priority — horizontal alignment
            if abs(dx) > 0.8:
                return 0 if dx > 0 else 2
            # Second priority — control descent speed
            if vy < -2.0:
                return 1  # THRUST_UP — too fast
            # Aligned and slow — let it land
            return 3  # NO_THRUST

        # ── ALIGN strategy ───────────────────────────────────────────
        # Priority 1 — horizontal correction
        if abs(dx) > 2.0:
            return 0 if dx > 0 else 2

        # Priority 2 — controlled descent
        if altitude > 6.0:
            # High up — fall freely but not too fast
            if vy < -3.0:
                return 1  # brake
            return 3  # NO_THRUST — fall

        elif altitude > 3.0:
            # Mid altitude — descend carefully
            if vy < -2.5:
                return 1  # brake
            if abs(dx) > 1.0:
                return 0 if dx > 0 else 2  # correct alignment
            return 3  # NO_THRUST

        else:
            # Low altitude — very careful
            if vy < -1.5:
                return 1  # brake hard
            if abs(dx) > 0.5:
                return 0 if dx > 0 else 2
            return 3  # NO_THRUST

    def reset(self):
        self.step_count = 0
        self.current_strategy = "ALIGN"
        self.reasoner.reset_history()