"""
BridgeSign – Sign Text Processor
=================================
Takes transcribed speech text and maps each word to sign language guidance.
For words that have a known gesture (from word_signs.py), shows the gesture tip.
For other words, instructs the user to fingerspell using the alphabet.
"""

from word_signs import SIGN_TIPS, ALL_WORD_SIGNS, display_name


# Build a lookup: lowercase word → (original_label, tip)
_WORD_LOOKUP = {}
for label in ALL_WORD_SIGNS:
    # "THANK_YOU" → lookup keys: "thank you", "thank_you", "thankyou"
    readable = display_name(label).lower()       # "thank you"
    _WORD_LOOKUP[readable] = label
    _WORD_LOOKUP[label.lower()] = label           # "thank_you"
    _WORD_LOOKUP[readable.replace(" ", "")] = label  # "thankyou"

def asl_gloss_order(text: str) -> str:
    """
    Applies basic English to ASL gloss reordering rules:
    - Drops articles and copulas
    - Moves time words to the front
    - Drops 'do/does/did' in questions
    """
    if not text or not text.strip():
        return ""
    
    words = text.strip().split()
    
    # Simple rule-based reordering lists
    articles = {"a", "an", "the"}
    copulas = {"is", "are", "am", "was", "were", "be", "been", "being"}
    time_words = {"tomorrow", "yesterday", "today", "now", "later"}
    aux_verbs = {"do", "does", "did"}
    
    filtered_words = []
    extracted_time_words = []
    
    for word in words:
        clean_word = word.lower().strip(".,!?;:'\"")
        
        # Move time words to the front
        if clean_word in time_words:
            extracted_time_words.append(word)
            continue
            
        # Drop articles, copulas, and auxiliary 'do/does/did'
        if clean_word in articles or clean_word in copulas or clean_word in aux_verbs:
            continue
            
        filtered_words.append(word)
        
    # Reassemble: Time words first, then the rest
    gloss_words = extracted_time_words + filtered_words
    
    # We could implement "not" after verb, or topic before comment here in the future
    return " ".join(gloss_words)

def process_speech_gloss(text: str) -> tuple:
    """
    First applies ASL gloss reordering, then processes the speech into guidance.
    Returns: (gloss_order_str, guidance_list)
    """
    gloss_str = asl_gloss_order(text)
    return gloss_str, process_speech(gloss_str)


def process_speech(text: str) -> list:
    """
    Convert a transcribed sentence into sign-language guidance.

    Parameters
    ----------
    text : str
        Raw transcribed text, e.g. "I need help please"

    Returns
    -------
    list of dict, each with keys:
        word      – the original word
        has_sign  – True if a known gesture exists
        tip       – how to sign it (gesture description or "Fingerspell: X")
        type      – "gesture" | "letter" | "fingerspell"
        sign_label– the canonical sign label (e.g. "HELP") or None
    """
    if not text or not text.strip():
        return []

    words = text.strip().split()
    result = []

    i = 0
    while i < len(words):
        word = words[i]
        word_lower = word.lower().strip(".,!?;:'\"")
        matched = False

        # Try two-word compound match first (e.g. "thank you" → THANK_YOU)
        if i + 1 < len(words):
            compound = word_lower + " " + words[i + 1].lower().strip(".,!?;:'\"")
            if compound in _WORD_LOOKUP:
                label = _WORD_LOOKUP[compound]
                tip = SIGN_TIPS.get(label, "Hold the sign naturally.")
                result.append({
                    "word": f"{word} {words[i + 1]}",
                    "has_sign": True,
                    "tip": tip,
                    "type": "gesture",
                    "sign_label": label,
                })
                i += 2
                matched = True

        if not matched:
            if word_lower in _WORD_LOOKUP:
                label = _WORD_LOOKUP[word_lower]
                tip = SIGN_TIPS.get(label, "Hold the sign naturally.")
                result.append({
                    "word": word,
                    "has_sign": True,
                    "tip": tip,
                    "type": "gesture",
                    "sign_label": label,
                })
            elif len(word_lower) == 1 and word_lower.isalpha():
                # Single letter
                result.append({
                    "word": word,
                    "has_sign": True,
                    "tip": f"Sign the letter '{word.upper()}' using the ASL alphabet.",
                    "type": "letter",
                    "sign_label": word.upper(),
                })
            else:
                # No known gesture — fingerspell it
                spelled = " - ".join(ch.upper() for ch in word_lower if ch.isalpha())
                result.append({
                    "word": word,
                    "has_sign": False,
                    "tip": f"Fingerspell: {spelled}" if spelled else "No sign available",
                    "type": "fingerspell",
                    "sign_label": None,
                })
            i += 1

    return result


def get_summary(guidance: list) -> dict:
    """
    Return a quick summary of the guidance list.
    """
    total = len(guidance)
    gestures = sum(1 for g in guidance if g["type"] == "gesture")
    fingerspell = sum(1 for g in guidance if g["type"] == "fingerspell")
    letters = sum(1 for g in guidance if g["type"] == "letter")
    return {
        "total_words": total,
        "gesture_count": gestures,
        "fingerspell_count": fingerspell,
        "letter_count": letters,
    }


if __name__ == "__main__":
    # Quick test
    test_sentences = [
        "I need help please",
        "Where is the doctor",
        "Hello my name is John",
        "Thank you very much",
        "I am hungry and cold",
        "Are you going tomorrow?",
        "I did not understand the homework."
    ]
    for sent in test_sentences:
        print(f"\n{'='*50}")
        print(f"  Speech: \"{sent}\"")
        gloss, guidance = process_speech_gloss(sent)
        print(f"  Gloss:  \"{gloss}\"")
        print(f"{'='*50}")
        for g in guidance:
            icon = "[SIGN]" if g["has_sign"] else "[SPELL]"
            print(f"  {icon} {g['word']:15s} -> {g['tip'][:60]}")
        summary = get_summary(guidance)
        print(f"  --- {summary['gesture_count']} gestures, "
              f"{summary['fingerspell_count']} fingerspell, "
              f"{summary['letter_count']} letters")

