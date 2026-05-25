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
""".lower().split())

BOUNDARY_SECONDS   = 0.55  # seconds of no-hand → flush word (shortened for faster response)
SENTENCE_PAUSE_SEC = 2.5   # seconds of no new word → flush sentence
MAX_BUFFER_LEN     = 22    # safeguard: auto-flush if buffer grows too long
WORD_DEDUP_SEC     = 0.8   # ignore the same gesture word if repeated within this window


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
        self._last_hand_ts = time.time()
        # Only append if different from the last letter (avoid stuttering)
        if not self._buf or self._buf[-1] != letter.upper():
            self._buf.append(letter.upper())
            if len(self._buf) > MAX_BUFFER_LEN:
                # Buffer overflow → flush immediately
                self._flush()

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

        normalized = self._normalize_word(word)
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
                self.sentence     = " ".join(self._words)

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
        if len(raw) == 1:
            return raw
        return self._validate(raw)

    def _flush(self) -> str:
        """Assemble and validate the current buffer, then clear it."""
        raw = "".join(self._buf)
        self._buf = []
        if not raw:
            return ""
        if len(raw) == 1:
            return raw  # single letter (I, A) always valid
        return self._validate(raw)

    def _commit_word(self, word: str, now: float | None = None):
        """Append a completed word to the sentence state."""
        if not word:
            return
        if now is None:
            now = time.time()
        self.last_word = word
        self._last_word_ts = now
        self._words.append(word)
        self.sentence = " ".join(self._words)

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
          4. Raw fallback — valid finger-spelled content not in dictionary
        """
        if raw.lower() in _KNOWN_WORDS:
            return raw
        deduped = self._dedup(raw)
        if deduped.lower() in _KNOWN_WORDS:
            return deduped
        expanded = self._best_expansion(deduped)
        if expanded and expanded.lower() in _KNOWN_WORDS:
            return expanded
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
