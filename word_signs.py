"""
BridgeSign – Word Sign Collection List
=======================================
Master list of target word signs to collect training data for.
Organised by priority: collect Phase 1 first, then Phase 2, then Phase 3.

Run collection with:
  python data_collector.py --sign HELP --type word --samples 200

Each sign needs 200+ samples from varied distances / lighting conditions.
"""

# ── Phase 1: Critical / Emergency ────────────────────────────────────────────
# These are the most important words — collect these FIRST.
PHASE_1 = [
    "HELP",
    "STOP",
    "PAIN",
    "DOCTOR",
    "WATER",
    "FOOD",
    "TOILET",
    "FIRE",
    "POLICE",
    "AMBULANCE",
    "EMERGENCY",
    "DANGER",
    "SICK",
    "HURT",
]

# ── Phase 2: Daily Communication ─────────────────────────────────────────────
# High-frequency signs that make up real conversations.
PHASE_2 = [
    "HELLO",
    "GOODBYE",
    "PLEASE",
    "THANK_YOU",      # underscore → displayed as "THANK YOU"
    "SORRY",
    "YES",
    "NO",
    "MAYBE",
    "AGAIN",
    "WAIT",
    "UNDERSTAND",
    "SLOW",
    "MORE",
    "LESS",
    "NEED",
    "WANT",
    "HAPPY",
    "SAD",
    "ANGRY",
    "TIRED",
]

# ── Phase 3: Questions & Directions ─────────────────────────────────────────
PHASE_3 = [
    "WHERE",
    "WHAT",
    "WHEN",
    "WHO",
    "WHY",
    "HOW",
    "LEFT",
    "RIGHT",
    "COME",
    "GO",
    "HERE",
    "THERE",
    "GOOD",
    "BAD",
    "HUNGRY",
    "COLD",
    "HOT",
    "LOVE",
    "FAMILY",
    "FRIEND",
    "NAME",
    "HOME",
    "WORK",
    "SCHOOL",
]

# ── All phases combined (for reference) ─────────────────────────────────────
ALL_WORD_SIGNS = PHASE_1 + PHASE_2 + PHASE_3

# ── Tips for each sign ───────────────────────────────────────────────────────
# Short description shown to the collector during data collection session.
SIGN_TIPS = {
    "HELP":       "Open hand, thumb up, place on flat palm — push upward together",
    "STOP":       "Flat hand horizontal, bring down sharply onto the other palm",
    "PAIN":       "Both index fingers pointing toward each other, twist slightly",
    "DOCTOR":     "Tap your fingertips on your wrist like taking a pulse",
    "WATER":      "W handshape — tap chin twice with index-middle-ring tips",
    "FOOD":       "Flat O handshape — tap lips twice",
    "TOILET":     "T handshape — shake slightly side to side",
    "FIRE":       "Wiggle fingers upward from low, like rising flames",
    "POLICE":     "C handshape — tap left chest where a badge would be",
    "AMBULANCE":  "A handshape — rotate in a circle (siren motion)",
    "EMERGENCY":  "E handshape — shake rapidly from side to side",
    "DANGER":     "A fist with thumb up — brush upward past the back of other hand",
    "SICK":       "Middle finger touches forehead, other middle touches stomach",
    "HURT":       "Both index fingers pointing at each other — twist toward each other",
    "HELLO":      "Open hand at temple — sweep outward in a salute wave",
    "GOODBYE":    "Open hand, wave fingers down then back up (like waving bye)",
    "PLEASE":     "Flat hand on chest — move in a circular motion",
    "THANK_YOU":  "Flat hand at chin — move forward toward the other person",
    "SORRY":      "A fist — rub in a circle on chest",
    "YES":        "S fist — nod it up and down like a yes nod",
    "NO":         "Index and middle finger snap down to touch thumb",
    "MAYBE":      "Both flat hands — alternate up and down like weighing scales",
    "AGAIN":      "Curved hand — tap the flat palm of the other hand",
    "WAIT":       "Both open hands facing up — wiggle fingers as if waiting",
    "UNDERSTAND": "Index finger at forehead — flick upward (like a lightbulb)",
    "SLOW":       "Drag one hand slowly up the back of the other forearm",
    "MORE":       "Flat O shape both hands — tap tips together twice",
    "LESS":       "Both flat hands — bring upper hand slightly down toward lower",
    "NEED":       "X handshape — bend downward repeatedly",
    "WANT":       "Both curved open hands — draw toward body",
    "HAPPY":      "Flat hand on chest — brush upward and out twice",
    "SAD":        "Both open hands in front of face — lower them slowly",
    "ANGRY":      "Claw hand in front of face — pull slightly inward",
    "TIRED":      "Both bent hands on chest — rotate forward and down (slumping)",
    "WHERE":      "Index finger points up — shake side to side",
    "WHAT":       "Index fingers together horizontally — wiggle side to side",
    "WHEN":       "Index finger circles down to touch the other index finger",
    "WHO":        "Circle around lips with index finger",
    "WHY":        "Touch forehead with fingers — bring hand down into Y shape",
    "HOW":        "Both curved hands together — rotate forward and upward",
    "LEFT":       "L handshape — point and move to the left",
    "RIGHT":      "R handshape — point and move to the right",
    "COME":       "Index finger curves up — draw toward body",
    "GO":         "Both index fingers together — move away from body",
    "HERE":       "Both flat hands circle horizontally in front of you",
    "THERE":      "Index finger points outward (to the side)",
    "GOOD":       "Flat hand at lips — move forward then down into palm",
    "BAD":        "Flat hand at lips — swipe down and away, flip to back",
    "HUNGRY":     "C handshape at throat — move downward",
    "COLD":       "Both A fists — shake near shoulders while hunching",
    "HOT":        "A handshake at mouth — sweep outward (like spitting heat)",
    "LOVE":       "Cross arms over chest (hug yourself)",
    "FAMILY":     "F handshape both hands — circle away from body outward",
    "FRIEND":     "Hook index fingers together — swap which is on top, twice",
    "NAME":       "H handshape — tap twice on the back of other H hand",
    "HOME":       "Flat O at mouth — move to cheek",
    "WORK":       "S fist taps on back of other S fist twice",
    "SCHOOL":     "Flat hand claps on palm twice (like a teacher clapping)",
}


def get_tip(sign: str) -> str:
    """Return the collection tip for a given sign label."""
    return SIGN_TIPS.get(sign, "Hold the sign naturally in front of the camera.")


def display_name(sign: str) -> str:
    """Convert THANK_YOU → THANK YOU for display."""
    return sign.replace("_", " ")


if __name__ == "__main__":
    print(f"\n{'='*52}")
    print(f"  BridgeSign Word Signs — {len(ALL_WORD_SIGNS)} total targets")
    print(f"{'='*52}")
    print(f"\n  Phase 1 ({len(PHASE_1)} signs — collect FIRST):")
    for s in PHASE_1:
        print(f"    • {display_name(s):<18} → {get_tip(s)[:55]}…")
    print(f"\n  Phase 2 ({len(PHASE_2)} signs):")
    for s in PHASE_2:
        print(f"    • {display_name(s):<18} → {get_tip(s)[:55]}…")
    print(f"\n  Phase 3 ({len(PHASE_3)} signs):")
    for s in PHASE_3:
        print(f"    • {display_name(s):<18} → {get_tip(s)[:55]}…")
    print()
