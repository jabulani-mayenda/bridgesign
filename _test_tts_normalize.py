from text_to_speech import TextToSpeech
t = TextToSpeech()

cases = [
    ("HELLO",         "Hello"),
    ("HELLO WORLD",   "Hello World"),
    ("GOOD MORNING",  "Good Morning"),
    ("I need help",   "I need help"),   # mixed case - should pass through
    ("THANK YOU",     "Thank You"),
    ("WORLD",         "World"),
    ("MY NAME IS",    "My Name Is"),
]

all_ok = True
for inp, expected in cases:
    result = t._normalize(inp)
    ok = result == expected
    tag = "OK  " if ok else "FAIL"
    print(f'{tag}  "{inp}" -> "{result}"  (expected "{expected}")')
    if not ok:
        all_ok = False

print()
print("ALL OK" if all_ok else "SOME TESTS FAILED")
