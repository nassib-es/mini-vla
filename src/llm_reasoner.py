import requests
import json


class LLMReasoner:
    """
    Sends drone state description to local Llama 3.1 via Ollama
    and gets back an action decision.
    
    This is the "brain" of the VLA system — it reasons about
    the scene and decides what the drone should do next.
    """

    SYSTEM_PROMPT = """You are a drone flight strategist. Choose ONE strategy:

- ALIGN: drone is NOT above the platform, needs horizontal correction
- LAND: drone IS above or very close to the platform, ready to land

Respond with ONLY: ALIGN or LAND"""

    def decode_strategy(self, response: str) -> str:
        """Extract strategy from LLM response."""
        response = response.upper().strip()
        for strategy in ["BRAKE", "LAND", "DESCEND", "ALIGN"]:
            if strategy in response:
                return strategy
        return None

    def __init__(self, model="llama3.1", host="http://localhost:11434"):
        self.model   = model
        self.host    = host
        self.url     = f"{host}/api/chat"

    def decide(self, scene_description: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": scene_description}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 20
            }
        }
        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"].strip()

    def reset_history(self):
        pass # kept for compatibility

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except:
            return False


if __name__ == '__main__':
    reasoner = LLMReasoner()

    if not reasoner.is_available():
        print("Ollama not running! Start it with: ollama serve")
        exit(1)

    print("=== LLM Reasoner Test ===\n")

    test_scene = """DRONE STATUS:
- Position: 2.0m to the RIGHT of the platform, high up at 8.0m above the platform
- Motion: descending at 2.0 m/s, drifting LEFT at 1.0 m/s
- Platform: moving RIGHT at 1.5 m/s
- Drone coords: (3.0, 9.0) | Platform: 1.0 | Velocity: (-1.0, -2.0)"""

    print("Scene:\n", test_scene)
    print("\nAsking LLM...")
    action = reasoner.decide(test_scene)
    print(f"\nLLM Decision: {action}")