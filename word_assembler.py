"""
BridgeSign – Word Assembler
===========================
Watches the stream of recognised letters and assembles them into words.

Word boundary  → hand disappears for >= BOUNDARY_SECONDS
Sentence boundary → no new word for >= SENTENCE_PAUSE_SEC

The assembler works in both modes:
  • letter mode  → tick() is still called but results are ignored by caller
  • word mode    → caller uses the returned dict to update the UI
"""

import time

# ── Compact dictionary for validation (no extra packages needed) ──────────────
# Includes common English words + the top sign-language vocabulary targets.
_KNOWN_WORDS = set("""
a i am are at be by do go he hi if in is it me my no of on or so to up us we
able add age ago aid aim air all and any arm art ask bad bag ban bar bat bay
bed big bit box boy bud bug bus but buy can car cat cop cow cry cup cut dad day
did dog dry due eat egg end era far fat few fly for fun gap gas get god got gun
gut had has hat hay her him his hit hop how hub hug hum ice ill its jam jar job
joy jug key kid lag lap law lay leg let lid lip log lot low mad man map mat may
mix mob mom mop mud mug nap net nod nor not now nut oak odd off oil old one opt
our out own pan pat pay pen pet pie pig pin pit pop pot pry pub put rag ran rap
rat raw ray red rep rid rip rob rod rot row rub run rut sad sat saw say sea see
set sew she shy sin sip sir sit six ski sky sly sob son spa spy sum sun sup tab
tan tap tar tax tea ten tip toe ton too top tug two use van vat via yup yes yet
you yum zip zoo able also area baby back ball band bank base bath bear beat been
best body bold bomb bond book born both bowl buck bulk busy call camp card care
cash cast chat city clap clay clip club coal coat code cold come cook cool copy
core corn cost crop cure cute damp dark dart data date dawn dead deal dear debt
deck deep deny desk diet dirt disk dock does done door dose down draw drop drug
drum dump dusk dust duty each earn ease east easy edge face fact fail fair fall
fame fast feed feel fell file fill film find fine fire firm fish fist flag flat
flew flip flow foam folk fond font food fool foot ford fork form fort four free
from fuel full fund gale game gang gate gave gaze gear gene gift glad glee glow
goal goat goes gold golf gone good grab gram gray grew grid grim grow gulf gull
gust hack hail hall halt hand hang hard harm hate have head heal hear heat held
hell help here hero hide high hill hire hold hole home hook hope horn host hour
huge hull hunt hurt idea idle inch into iron item jail jazz jerk join joke jump
just keen keep kill kind king know lace lack laid lake land lane last late lead
leaf lean left lend less lift like lily lime line link lion list live load lock
long look lord lose loud love luck lump lung made mail main make male mall mama
many mark mask mass math meal mean meet melt memo menu mesh mild mile mill mind
mine mint miss mode mold moon more most move much mule muse must nail name navy
near neck need next nice nine node none noon norm nose note noun nail odds once
only open oral over pace pack paid pain pair pale palm park part pass past path
peak peel peer pest pick pile pine pink pint pipe plan play plod plot plug poem
poet pole poll pond poor port pose post pour pray prey pure push rack rage rain
race ramp rare rate read real rent rest rice rich ride ring rise risk road rock
role roll roof room rope rose rows rush rust safe sail salt same sand save scan
self send ship shop shop show sick side sign silk sing sink size skip slow slug
snap snow soft soil sole some sort soul soup sour span spin spot spray star stay
step stop such suit swap swim tall tape task team tear tell tent test text than
then they thin this tier till time tiny tire told tone torn tow town tray tree
true tube tune turn twin type uncle upon used vain vast very view visit voice
woke walk wall warn wart wash wave weak wear week well went west wide wife wild
will wind wing wipe wire wish wolf wood wool word wore work worm wrap year zero
HELP WATER YES NO PLEASE THANK STOP DOCTOR PAIN SORRY FIRE POLICE AMBULANCE
DANGER EMERGENCY HAPPY SAD ANGRY SCARED TIRED WHERE WHAT WHEN WHO HOW WHY LEFT
RIGHT COME GO HELLO GOODBYE MORNING NIGHT FOOD DRINK TOILET NEED WANT UNDERSTAND
AGAIN SLOW SPEAK MORE LESS WAIT HERE THERE TOGETHER ALONE FAMILY FRIEND SCHOOL
WORK HOME NAME AGE SICK HUNGRY COLD HOT LOVE GOOD BAD BEAUTIFUL
BENSON
""".lower().split())

WORD_PAUSE_THRESHOLD_MS = 400
BOUNDARY_SECONDS   = WORD_PAUSE_THRESHOLD_MS / 1000.0
SENTENCE_PAUSE_SEC = 8.0   # keep completed phrases visible long enough for demos/conversation
MAX_BUFFER_LEN     = 22    # safeguard: auto-flush if buffer grows too long
WORD_DEDUP_SEC     = 0.8   # ignore the same gesture word if repeated within this window
LETTER_DEDUP_SEC   = 0.85  # absorbs held-sign/tracking stutter; doubles are recovered on validation

CONFUSABLE_LETTERS = {
    "C": "O",
    "O": "C",
    "D": "L",
    "L": "D",
}

SIGN_WORD_MAP = {
    "I": "I",
    "A": "A",
    "LOVE": "LOVE",
    "YOU": "YOU",
    "ILOVEYOU": "I LOVE YOU",
    "THANKYOU": "THANK YOU",
    "THANK YOU": "THANK YOU",
    "HELLO": "HELLO",
    "HELP": "HELP",
    "YES": "YES",
    "NO": "NO",
    "PLEASE": "PLEASE",
    "SORRY": "SORRY",
    "STOP": "STOP",
}

PHRASE_DISPLAY_MAP = {
    "HI MY NAME IS BENSON": "Hi, my name is Benson.",
    "HELLO MY NAME IS BENSON": "Hi, my name is Benson.",
    "MY NAME IS BENSON": "My name is Benson.",
    "I LOVE YOU": "I love you.",
}


class WordAssembler:
    """
    Push letters in via push_letter().
    Call tick() on every inference cycle (hand_present tells it if hand is visible).
    Read the returned dict to update UI state.
    """

    def __init__(self):
        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def push_letter(self, letter: str):
        """Call when a letter is confidently recognised."""
        now = time.time()
        self._last_hand_ts = now
        clean = str(letter or "").strip().upper()
        if not clean:
            return

        if clean == self._last_letter:
            elapsed = now - (self._last_letter_ts or 0.0)
            if elapsed < LETTER_DEDUP_SEC:
                return

        self._buf.append(clean)
        self._last_letter = clean
        self._last_letter_ts = now
        print(f"[WordModule] Buffered: \"{''.join(self._buf)}\" (pause: false)")
        if len(self._buf) > MAX_BUFFER_LEN:
            completed_word = self._flush()
            if completed_word:
                self._commit_word(completed_word, now)

    def push_word(self, word: str) -> str:
        """
        Commit a fully recognised gesture word immediately.

        If there are pending finger-spelled letters in the buffer, flush them
        first so word-level gestures do not get concatenated into the letter
        stream.

        Deduplication: if the exact same word was committed within
        WORD_DEDUP_SEC, it is ignored to stop repeated gesture emissions.
        """
        now = time.time()
        self._last_hand_ts = now

        if self._buf:
            buffered_word = self._flush()
            if buffered_word:
                self._commit_word(buffered_word, now)

        normalized = self._map_word(self._normalize_word(word))
        if not normalized:
            return ""

        # Word-level deduplication: skip if same word very recently committed
        if (normalized == self.last_word
                and self._last_word_ts is not None
                and (now - self._last_word_ts) < WORD_DEDUP_SEC):
            return normalized

        self._commit_word(normalized, now)
        return normalized

    def tick(self, hand_present: bool) -> dict:
        """
        Call on every inference cycle regardless of whether a letter was found.

        Returns:
            dict with keys:
              word_buffer       – letters being assembled (string)
              last_word         – most recently completed word
              sentence          – words accumulated so far
              completed_word    – non-None only in the frame a word is finalised
              completed_sentence– non-None only in the frame a sentence is finalised
        """
        now = time.time()
        completed_word     = None
        completed_sentence = None

        if hand_present:
            self._last_hand_ts = now  # keep refreshing while hand is visible

        # ── Word boundary ──────────────────────────────────────────────────
        if (self._buf
                and self._last_hand_ts is not None
                and not hand_present
                and (now - self._last_hand_ts) >= BOUNDARY_SECONDS):
            completed_word = self._flush()
            if completed_word:
                self.last_word    = completed_word
                self._last_word_ts = now
                self._words.append(completed_word)
                self.sentence     = self._format_sentence(self._words)
                print(f"[WordModule] Pause detected ({int((now - self._last_hand_ts) * 1000)}ms) -> confirmed word: \"{completed_word}\"")
                print(f"[WordModule] Final phrase: \"{self.sentence}\"")

        # ── Sentence boundary ──────────────────────────────────────────────
        if (self._words
                and not self._buf
                and self._last_word_ts is not None
                and (now - self._last_word_ts) >= SENTENCE_PAUSE_SEC):
            completed_sentence = self.sentence
            self._words        = []
            self.sentence      = ""
            self._last_word_ts = None

        return {
            "word_buffer":        "".join(self._buf),
            "last_word":          self.last_word,
            "sentence":           self.sentence,
            "completed_word":     completed_word,
            "completed_sentence": completed_sentence,
        }

    def reset(self):
        """Full reset (called when camera stops or mode switches)."""
        self._buf          = []
        self._words        = []
        self._last_hand_ts = None
        self._last_word_ts = None
        self.last_word     = ""
        self.sentence      = ""
        self._last_letter = ""
        self._last_letter_ts = None

    def set_phrase(self, phrase: str) -> str:
        """Replace the active sentence with a complete display phrase."""
        words = [self._map_word(part) for part in str(phrase or "").strip().split() if part.strip()]
        self._buf = []
        self._words = words
        self.last_word = words[-1] if words else ""
        self._last_word_ts = time.time() if words else None
        self.sentence = self._format_sentence(words)
        return self.sentence

    def has_pending_letters(self) -> bool:
        """True while the user is actively building a finger-spelled word."""
        return bool(self._buf)

    # ── Internals ─────────────────────────────────────────────────────────────

    def manual_flush(self) -> str:
        """
        Immediately commit the current buffer as a word — called when the
        user presses the 'Next Word' button, no need to wait for timeout.
        """
        if not self._buf:
            return ""
        now = time.time()
        completed_word = self._flush()
        if completed_word:
            self._last_hand_ts = now
            self._commit_word(completed_word, now)
        return completed_word

    def live_prediction(self) -> str:
        """
        Returns the best-guess word for the letters typed so far,
        without consuming the buffer. Used to show a live preview.
        """
        if not self._buf:
            return ""
        raw = "".join(self._buf)
        return self._map_word(raw if len(raw) == 1 else self._validate(raw))

    def _flush(self) -> str:
        """Assemble and validate the current buffer, then clear it."""
        raw = "".join(self._buf)
        self._buf = []
        if not raw:
            return ""
        if len(raw) == 1:
            return self._map_word(raw)
        return self._map_word(self._validate(raw))

    def _commit_word(self, word: str, now: float | None = None):
        """Append a completed word to the sentence state."""
        if not word:
            return
        if now is None:
            now = time.time()
        self.last_word = word
        self._last_word_ts = now
        self._words.append(word)
        self.sentence = self._format_sentence(self._words)
        print(f"[WordModule] Final phrase: \"{self.sentence}\"")

    def undo_last_word(self) -> str:
        """Remove and return the most recently committed word."""
        if not self._words:
            return ""
        removed = self._words.pop()
        self.sentence = self._format_sentence(self._words)
        self.last_word = self._words[-1] if self._words else ""
        self._last_word_ts = time.time() if self._words else None
        return removed

    def clear_current_word(self):
        """Clear the active letter buffer but keep committed phrase words."""
        self._buf = []
        self._last_letter = ""
        self._last_letter_ts = None

    @staticmethod
    def _map_word(word: str) -> str:
        clean = str(word or "").replace("_", " ").strip()
        compact = clean.replace(" ", "").upper()
        return SIGN_WORD_MAP.get(clean.upper(), SIGN_WORD_MAP.get(compact, clean))

    @staticmethod
    def _format_sentence(words) -> str:
        raw = " ".join(str(word or "").strip() for word in words if str(word or "").strip())
        compact = " ".join(raw.replace("_", " ").upper().split())
        if compact in PHRASE_DISPLAY_MAP:
            return PHRASE_DISPLAY_MAP[compact]
        return raw

    @staticmethod
    def _normalize_word(word: str) -> str:
        """Convert storage labels like THANK_YOU into display text 'THANK YOU'.
        The words are stored with spaces in the sentence (joined by ' '.join)
        so each token stays as a single item in self._words.
        """
        return str(word or "").replace("_", " ").strip().upper()

    def _validate(self, raw: str) -> str:
        """
        Return the best matching known word, or the raw assembled string.

        Tries in order:
          1. Exact match (raw as typed)
          2. Full dedup  (HELLLO → HELLO)
          3. Selective expansion — insert one repeat at each position and
             check the dictionary  (recovers HELO → HELLO, WORL → WORLD …)
          4. Confusable correction — swap one/two lookalike letters
             (recovers HEDP → HELP, CCLD → COLD)
          5. Raw fallback — valid finger-spelled content not in dictionary
        """
        if raw.lower() in _KNOWN_WORDS:
            return raw
        deduped = self._dedup(raw)
        if deduped.lower() in _KNOWN_WORDS:
            return deduped
        expanded = self._best_expansion(deduped)
        if expanded and expanded.lower() in _KNOWN_WORDS:
            return expanded
        corrected = self._best_confusable_correction(deduped)
        if corrected and corrected.lower() in _KNOWN_WORDS:
            return corrected
        # Return raw — not in dictionary but still valid finger-spelled content
        return raw

    @staticmethod
    def _dedup(s: str) -> str:
        """Collapse runs of 3+ identical consecutive chars (HELLLLO→HELLO).
        Runs of exactly 2 are kept so intentional doubles survive."""
        if not s:
            return s
        out = [s[0]]
        run = 1
        for ch in s[1:]:
            if ch == out[-1]:
                run += 1
                if run <= 2:      # allow at most 2 of the same char in a row
                    out.append(ch)
            else:
                run = 1
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _best_expansion(s: str) -> str:
        """
        Try inserting one repeated character at each position of *s*.
        Returns the first expansion that lands in _KNOWN_WORDS, or "" if none.
        This recovers words like HELO → HELLO and WORL → WORLD.
        """
        for i, ch in enumerate(s):
            candidate = s[:i] + ch + s[i:]   # repeat s[i] once
            if candidate.lower() in _KNOWN_WORDS:
                return candidate
        return ""

    @staticmethod
    def _best_confusable_correction(s: str) -> str:
        """
        Try swapping visually similar letters. We prefer a single correction,
        then allow two corrections for short dictionary words where two close
        hand shapes were both misread.
        """
        if not s:
            return ""

        chars = list(s)
        positions = [i for i, ch in enumerate(chars) if ch in CONFUSABLE_LETTERS]

        for i in positions:
            ch = chars[i]
            replacement = CONFUSABLE_LETTERS.get(ch)
            candidate = chars.copy()
            candidate[i] = replacement
            candidate = "".join(candidate)
            if candidate.lower() in _KNOWN_WORDS:
                return candidate

        for pos_index, i in enumerate(positions):
            for j in positions[pos_index + 1:]:
                candidate_chars = chars.copy()
                candidate_chars[i] = CONFUSABLE_LETTERS[candidate_chars[i]]
                candidate_chars[j] = CONFUSABLE_LETTERS[candidate_chars[j]]
                candidate = "".join(candidate_chars)
                if candidate.lower() in _KNOWN_WORDS:
                    return candidate
        return ""
