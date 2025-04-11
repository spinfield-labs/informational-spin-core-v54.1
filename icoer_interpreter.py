# icoer_interpreter.py

def interpret_i_tgu(score: float) -> str:
    if score < 0.30:
        return "Low coherence: The response shows fragmentation, contradiction, or lack of structure."
    elif 0.30 <= score <= 0.60:
        return "Moderate coherence: Partial alignment across dimensions, but potential inconsistencies."
    else:
        return "High coherence: Harmonious balance between semantics, logic, emotion, and structure."

def symbolic_signature(score: float) -> str:
    if score > 0.85:
        return "Resonant Core: The message pulses with informational harmony."
    elif score > 0.60:
        return "Coherent Flow: The spin is active, balance is near."
    elif score > 0.40:
        return "Fragmented Echo: Structure exists, but dissonance remains."
    else:
        return "Chaotic Field: Truth cannot emerge without coherence."

# Example
if __name__ == "__main__":
    test_score = 0.73
    print(interpret_i_tgu(test_score))
    print(symbolic_signature(test_score))
