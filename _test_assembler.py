"""Integration test for WordAssembler — Windows-safe (no non-ASCII)."""
import sys, time
sys.path.insert(0, '.')
from word_assembler import WordAssembler

errors = 0

def flush_word(a):
    a._last_hand_ts = time.time() - 2.0
    return a.tick(hand_present=False)["completed_word"]

def flush_sentence(a):
    a._last_word_ts = time.time() - 9.0
    return a.tick(hand_present=False)["completed_sentence"]

def check(label, result, expected):
    global errors
    ok = result == expected
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}: got '{result}'  (expected '{expected}')")
    if not ok:
        errors += 1

# Test 1: HELLO (double-L gets eaten by push_letter dedup, recovered by _best_expansion)
a = WordAssembler()
for ch in "HELLO":
    a.push_letter(ch)
check("HELLO recovery  ", flush_word(a), "HELLO")

# Test 2: WORLD (clean word)
a2 = WordAssembler()
for ch in "WORLD":
    a2.push_letter(ch)
check("WORLD straight  ", flush_word(a2), "WORLD")

# Test 3: Full sentence "HI THERE"
a3 = WordAssembler()
for word in ["HI", "THERE"]:
    for ch in word:
        a3.push_letter(ch)
    a3._last_hand_ts = time.time() - 2.0
    a3.tick(hand_present=False)
check("sentence HI THERE", flush_sentence(a3), "HI THERE")

# Test 4: Triple camera stutter HELLLLO -> HELLO (dedup collapses runs > 2)
a4 = WordAssembler()
a4._buf = list("HELLLLO")
check("HELLLLO dedup   ", flush_word(a4), "HELLO")

# Test 5: Word not in dictionary stays as-is (finger-spelled content)
a5 = WordAssembler()
for ch in "JOHN":
    a5.push_letter(ch)
check("JOHN fallback   ", flush_word(a5), "JOHN")

# Test 6: HELP (emergency vocab word)
a6 = WordAssembler()
for ch in "HELP":
    a6.push_letter(ch)
check("HELP dict match ", flush_word(a6), "HELP")

# Test 7: Gesture word should commit directly without corrupting letter buffer
a7 = WordAssembler()
for ch in "HI":
    a7.push_letter(ch)
a7.push_word("THANK_YOU")
check("gesture word flush", a7.last_word, "THANK YOU")
check("mixed sentence    ", a7.sentence, "HI THANK YOU")

# Test 8: Word labels with underscores should be normalised for display
a8 = WordAssembler()
check("underscore normal ", a8.push_word("GOODBYE"), "GOODBYE")
check("space normalise   ", a8.push_word("THANK_YOU"), "THANK YOU")

# Test 9: Presentation intro phrase should display naturally
a9 = WordAssembler()
for word in ["HI", "MY", "NAME", "IS", "BENSON"]:
    a9.push_word(word)
check("intro sentence   ", a9.sentence, "Hi, my name is Benson.")

print()
if errors == 0:
    print("ALL TESTS PASSED")
else:
    print(f"{errors} TEST(S) FAILED")
    sys.exit(1)
