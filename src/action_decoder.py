class ActionDecoder:
    """
    Converts LLM text output to environment action integer.
    
    The LLM outputs natural language like "THRUST_LEFT"
    We map that to the integer the drone environment expects.
    
    Action space (matching drone_env.py):
        0 = thrust left
        1 = thrust up  
        2 = thrust right
        3 = no thrust
    """

    ACTION_MAP = {
        "THRUST_LEFT":  0,
        "THRUST_UP":    1,
        "THRUST_RIGHT": 2,
        "NO_THRUST":    3,
    }

    DEFAULT_ACTION = 3  # no thrust if parsing fails

    def decode(self, llm_response: str) -> int:
        """
        Parse LLM response and return action integer.
        Handles variations in LLM output format.
        """
        response = llm_response.upper().strip()

        # Direct match
        for key, value in self.ACTION_MAP.items():
            if key in response:
                return value

        # Fallback keyword matching
        if "LEFT" in response:
            return 0
        if "UP" in response:
            return 1
        if "RIGHT" in response:
            return 2
        if "NO" in response or "NOTHING" in response or "NONE" in response:
            return 3

        print(f"  Warning: could not parse '{llm_response}', defaulting to NO_THRUST")
        return self.DEFAULT_ACTION

    def action_name(self, action: int) -> str:
        """Return human readable action name."""
        names = {0: "THRUST_LEFT", 1: "THRUST_UP",
                 2: "THRUST_RIGHT", 3: "NO_THRUST"}
        return names.get(action, "UNKNOWN")


if __name__ == '__main__':
    decoder = ActionDecoder()

    print("=== Action Decoder Test ===\n")
    tests = [
        "THRUST_LEFT",
        "THRUST_UP",
        "thrust_right",
        "NO_THRUST",
        "I think the drone should go LEFT",
        "Move right to correct position",
        "gibberish response xyz",
    ]

    for t in tests:
        action = decoder.decode(t)
        print(f"  '{t}' → {action} ({decoder.action_name(action)})")