import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import cv2
from PIL import Image, ImageTk
import config
from camera import Camera
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from classifier import Classifier
from text_to_speech import TextToSpeech
from image_translator import ImageTranslator
from session_tracker import SessionTracker
from emergency_phrases import EmergencyPhrases
from learning_mode import LearningMode
from word_assembler import WordAssembler

# ── Colour Palette ──────────────────────────────────────────
BG        = "#0d0d0d"
PANEL     = "#141414"
CARD      = "#1a1a1a"
ACCENT    = "#ff6b00"
ACCENT2   = "#ff9933"
BLUE      = "#00c2ff"
GREEN     = "#00e676"
RED       = "#ff1744"
MUTED     = "#555555"
TEXT      = "#f5f5f5"
SUBTEXT   = "#aaaaaa"
FONT_HEAD = ("Segoe UI", 22, "bold")
FONT_SUB  = ("Segoe UI", 12)
FONT_MONO = ("Consolas", 13)
FONT_BIG  = ("Segoe UI", 32, "bold")

def make_btn(parent, text, cmd, color=ACCENT, width=18):
    btn = tk.Button(
        parent, text=text, command=cmd,
        bg=color, fg=TEXT, activebackground=ACCENT2,
        activeforeground=TEXT, relief="flat", cursor="hand2",
        font=("Segoe UI", 11, "bold"), padx=12, pady=8,
        width=width, bd=0
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT2))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn

class BridgeSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BridgeSign – Sign Language Translator")
        self.root.geometry("1100x680")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # core modules
        self.image_translator = ImageTranslator()
        self.tracker          = SessionTracker()
        self.tts              = TextToSpeech()
        self.emergency        = EmergencyPhrases()
        self.learning         = LearningMode()

        # state
        self.camera_running  = False
        self.learn_running   = False
        self.current_sign    = tk.StringVar(value="Hello")
        self._tk_img         = None
        self._assembler      = None  # created fresh in _run_pipeline

        # Word / sentence display vars
        self._buf_var  = tk.StringVar(value="–")   # live letter buffer
        self._pred_var = tk.StringVar(value="")    # live word prediction
        self._word_var = tk.StringVar(value="–")   # last completed word
        self._sent_var = tk.StringVar(value="")    # full sentence so far

        self._build_ui()

    # ── UI Construction ──────────────────────────────────────
    def _build_ui(self):
        self._build_header()

        # notebook / tabs
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook",        background=BG,    borderwidth=0)
        style.configure("Dark.TNotebook.Tab",    background=PANEL, foreground=SUBTEXT,
                         font=("Segoe UI", 11, "bold"), padding=[18, 8])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", ACCENT)])

        nb = ttk.Notebook(self.root, style="Dark.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self._tab_live(nb)
        self._tab_image(nb)
        self._tab_learn(nb)
        self._tab_emergency(nb)
        self._tab_stats(nb)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=PANEL, height=60)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⬡  BridgeSign", font=FONT_HEAD,
                 bg=PANEL, fg=ACCENT).pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(hdr, text="Real-time Sign Language Translator",
                 font=FONT_SUB, bg=PANEL, fg=SUBTEXT).pack(side=tk.LEFT, padx=4, pady=10)

        self._status_var = tk.StringVar(value="● Ready")
        tk.Label(hdr, textvariable=self._status_var, font=("Segoe UI", 11),
                 bg=PANEL, fg=GREEN).pack(side=tk.RIGHT, padx=20)

    # ── Tab 1 – Live Camera ──────────────────────────────────
    def _tab_live(self, nb):
        frame = tk.Frame(nb, bg=BG)
        nb.add(frame, text="  📷  Live Camera  ")

        # left – video
        left = tk.Frame(frame, bg=CARD, bd=0)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12,6), pady=12)

        self._video_lbl = tk.Label(left, bg="#000000", text="Camera Off",
                                   font=("Segoe UI", 16), fg=MUTED)
        self._video_lbl.pack(fill=tk.BOTH, expand=True)

        # right – controls
        right = tk.Frame(frame, bg=CARD, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6,12), pady=12)
        right.pack_propagate(False)

        # ── Tier 1: Live letter buffer ───────────────────────
        buf_card = tk.Frame(right, bg="#111111", bd=0)
        buf_card.pack(fill=tk.X, padx=12, pady=(14, 2))
        tk.Label(buf_card, text="SIGNING…", font=("Segoe UI", 8, "bold"),
                 bg="#111111", fg=MUTED).pack(pady=(8, 0))
        tk.Label(buf_card, textvariable=self._buf_var,
                 font=("Consolas", 20, "bold"), bg="#111111", fg=ACCENT,
                 wraplength=240, justify="center").pack(pady=(2, 4))
        # Live prediction shown beneath the raw letters
        tk.Label(buf_card, textvariable=self._pred_var,
                 font=("Segoe UI", 10, "italic"), bg="#111111", fg=ACCENT2,
                 wraplength=240, justify="center").pack(pady=(0, 8))

        # ── Tier 2: Last completed word ──────────────────────
        word_card = tk.Frame(right, bg="#0d1a2a", bd=0)
        word_card.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(word_card, text="LAST WORD", font=("Segoe UI", 8, "bold"),
                 bg="#0d1a2a", fg=MUTED).pack(pady=(8, 0))
        tk.Label(word_card, textvariable=self._word_var,
                 font=FONT_BIG, bg="#0d1a2a", fg=BLUE,
                 wraplength=240, justify="center").pack(pady=(2, 8))

        # ── Tier 3: Full sentence ────────────────────────────
        sent_card = tk.Frame(right, bg="#0a1f0a", bd=0)
        sent_card.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(sent_card, text="SENTENCE", font=("Segoe UI", 8, "bold"),
                 bg="#0a1f0a", fg=MUTED).pack(pady=(8, 0))
        tk.Label(sent_card, textvariable=self._sent_var,
                 font=("Segoe UI", 14, "bold"), bg="#0a1f0a", fg=GREEN,
                 wraplength=240, justify="center").pack(pady=(2, 8))

        ttk.Separator(right).pack(fill=tk.X, padx=12, pady=6)

        self._cam_btn = make_btn(right, "▶  Start Camera", self.toggle_camera, color=GREEN, width=24)
        self._cam_btn.pack(padx=12, pady=3)

        # Word control buttons
        btn_row = tk.Frame(right, bg=CARD)
        btn_row.pack(fill=tk.X, padx=12, pady=3)
        make_btn(btn_row, "⏎ Next Word",   self._next_word,   color=ACCENT, width=11).pack(side=tk.LEFT, padx=(0,4))
        make_btn(btn_row, "⌫ Undo Letter", self._undo_letter, color=MUTED,  width=11).pack(side=tk.LEFT)

        make_btn(right, "🔊  Speak Sentence", self._speak_sentence, color=BLUE, width=24).pack(padx=12, pady=3)
        make_btn(right, "🗑  Clear All",       self._clear_all,      color=MUTED, width=24).pack(padx=12, pady=3)

        ttk.Separator(right).pack(fill=tk.X, padx=12, pady=6)

        tk.Label(right, text="Event Log", font=("Segoe UI", 10, "bold"),
                 bg=CARD, fg=SUBTEXT).pack(padx=12, anchor="w")

        log_frame = tk.Frame(right, bg=CARD)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4,12))

        self._log = tk.Text(log_frame, bg="#111111", fg=TEXT, font=FONT_MONO,
                            relief="flat", state="disabled", wrap="word")
        self._log.pack(fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_frame, command=self._log.yview)
        self._log['yscrollcommand'] = sb.set

    def _log_append(self, text):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _next_word(self):
        """Immediately commit current buffer as a word (no need to wait for timeout)."""
        if self._assembler is not None:
            word = self._assembler.manual_flush()
            if word:
                self._buf_var.set("–")
                self._pred_var.set("")
                self._word_var.set(word)
                self._sent_var.set(self._assembler.sentence)
                self._log_append(f"✔ Word: {word}")
                self.tts.speak_async(word)

    def _undo_letter(self):
        """Remove the last letter from the assembler buffer."""
        if self._assembler is not None and self._assembler._buf:
            self._assembler._buf.pop()
            buf = self._assembler._buf
            self._buf_var.set("  ".join(buf) if buf else "–")
            pred = self._assembler.live_prediction()
            self._pred_var.set(f"→ {pred}" if pred else "")

    def _speak_sentence(self):
        """Speak the current sentence (or last word as fallback)."""
        sent = self._sent_var.get().strip()
        word = self._word_var.get().strip()
        val  = sent if sent else word
        if val and val != "–":
            self.tts.speak_async(val)

    def _clear_all(self):
        """Reset assembler state and clear all display widgets."""
        if self._assembler is not None:
            self._assembler.reset()
        self._buf_var.set("–")
        self._word_var.set("–")
        self._sent_var.set("")
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ── Tab 2 – Image ────────────────────────────────────────
    def _tab_image(self, nb):
        frame = tk.Frame(nb, bg=BG)
        nb.add(frame, text="  🖼  Image Translate  ")

        left = tk.Frame(frame, bg=CARD)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12,6), pady=12)

        self._img_lbl = tk.Label(left, bg="#000000", text="Upload an image to translate",
                                  font=("Segoe UI", 14), fg=MUTED)
        self._img_lbl.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(frame, bg=CARD, width=260)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6,12), pady=12)
        right.pack_propagate(False)

        self._img_result = tk.StringVar(value="–")
        res_card = tk.Frame(right, bg="#111111")
        res_card.pack(fill=tk.X, padx=12, pady=18)
        tk.Label(res_card, text="RESULT", font=("Segoe UI", 9),
                 bg="#111111", fg=MUTED).pack(pady=(12,0))
        tk.Label(res_card, textvariable=self._img_result, font=FONT_BIG,
                 bg="#111111", fg=ACCENT2).pack(pady=(0,12))

        make_btn(right, "📂  Upload Image", self.upload_image, color=ACCENT,  width=22).pack(padx=12, pady=8)
        make_btn(right, "🔊  Speak Result", self._speak_img,  color=BLUE,    width=22).pack(padx=12, pady=6)

    def _speak_img(self):
        val = self._img_result.get()
        if val and val != "–":
            self.tts.speak_async(val)

    # ── Tab 3 – Learning Mode ────────────────────────────────
    def _tab_learn(self, nb):
        frame = tk.Frame(nb, bg=BG)
        nb.add(frame, text="  🎓  Learning Mode  ")

        left = tk.Frame(frame, bg=CARD)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12,6), pady=12)

        self._learn_lbl = tk.Label(left, bg="#000000", text="Learning Mode Paused",
                                   font=("Segoe UI", 14), fg=MUTED)
        self._learn_lbl.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(frame, bg=CARD, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6,12), pady=12)
        right.pack_propagate(False)

        tk.Label(right, text="Choose a Sign to Practice",
                 font=("Segoe UI", 12, "bold"), bg=CARD, fg=TEXT).pack(padx=12, pady=(18,6))

        signs = ["Hello","Thank you","I love you","Yes","No"]
        sign_menu = tk.OptionMenu(right, self.current_sign, *signs)
        sign_menu.config(bg=CARD, fg=TEXT, font=FONT_SUB, relief="flat",
                         highlightthickness=0, activebackground=ACCENT)
        sign_menu["menu"].config(bg=PANEL, fg=TEXT)
        sign_menu.pack(fill=tk.X, padx=12, pady=4)

        self._tip_var = tk.StringVar(value="Select a sign and start the camera.")
        tip_lbl = tk.Label(right, textvariable=self._tip_var, font=("Segoe UI", 10),
                           bg=CARD, fg=SUBTEXT, wraplength=230, justify="left")
        tip_lbl.pack(padx=12, pady=8, anchor="w")

        self._learn_btn = make_btn(right, "▶  Start Practice", self.toggle_learn, color=GREEN, width=22)
        self._learn_btn.pack(padx=12, pady=8)

        self._fb_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self._fb_var, font=("Segoe UI", 11, "bold"),
                 bg=CARD, fg=BLUE, wraplength=230).pack(padx=12, pady=4)

    # ── Tab 4 – Emergency Phrases ────────────────────────────
    def _tab_emergency(self, nb):
        frame = tk.Frame(nb, bg=BG)
        nb.add(frame, text="  🚨  Emergency  ")

        tk.Label(frame, text="Quick-Access Emergency Phrases",
                 font=("Segoe UI", 16, "bold"), bg=BG, fg=RED).pack(pady=(20,4))
        tk.Label(frame, text="Tap a phrase to instantly speak it aloud.",
                 font=FONT_SUB, bg=BG, fg=SUBTEXT).pack(pady=(0,16))

        phrases = self.emergency.get_phrases()
        grid = tk.Frame(frame, bg=BG)
        grid.pack(expand=True)

        for i, (pid, phrase) in enumerate(phrases.items()):
            row, col = divmod(i, 2)
            card = tk.Frame(grid, bg=CARD, padx=20, pady=16)
            card.grid(row=row, column=col, padx=12, pady=10, sticky="ew")
            tk.Label(card, text=phrase, font=("Segoe UI", 12),
                     bg=CARD, fg=TEXT, wraplength=320, justify="left").pack(anchor="w")
            btn = make_btn(card, "🔊  Speak", lambda p=pid: self.emergency.play_phrase(p),
                           color=RED, width=12)
            btn.pack(anchor="e", pady=(8,0))

    # ── Tab 5 – Stats ────────────────────────────────────────
    def _tab_stats(self, nb):
        frame = tk.Frame(nb, bg=BG)
        nb.add(frame, text="  📊  Stats  ")

        top = tk.Frame(frame, bg=BG)
        top.pack(fill=tk.X, padx=20, pady=16)
        tk.Label(top, text="Session Statistics", font=("Segoe UI", 16, "bold"),
                 bg=BG, fg=TEXT).pack(side=tk.LEFT)
        make_btn(top, "↺  Refresh", self._refresh_stats, color=BLUE, width=12).pack(side=tk.RIGHT)

        self._stats_frame = tk.Frame(frame, bg=BG)
        self._stats_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        self._refresh_stats()

    def _refresh_stats(self):
        for w in self._stats_frame.winfo_children():
            w.destroy()

        stats = self.tracker.get_stats()
        cards = [
            ("Total Translations", stats["total"], ACCENT),
            ("Most Common Sign",   stats["most_common"], BLUE),
        ]
        for label, val, color in cards:
            card = tk.Frame(self._stats_frame, bg=CARD, padx=24, pady=20)
            card.pack(side=tk.LEFT, padx=10, pady=10)
            tk.Label(card, text=str(val), font=("Segoe UI", 28, "bold"),
                     bg=CARD, fg=color).pack()
            tk.Label(card, text=label, font=("Segoe UI", 10),
                     bg=CARD, fg=SUBTEXT).pack()

        if stats.get("counts"):
            tk.Label(self._stats_frame, text="Sign Breakdown",
                     font=("Segoe UI", 12, "bold"), bg=BG, fg=TEXT).pack(
                     anchor="w", pady=(16,4))
            for lbl, count in stats["counts"].items():
                row = tk.Frame(self._stats_frame, bg=CARD)
                row.pack(fill=tk.X, pady=2)
                tk.Label(row, text=f"  {lbl}", font=FONT_SUB, bg=CARD, fg=TEXT,
                         width=20, anchor="w").pack(side=tk.LEFT)
                tk.Label(row, text=str(count), font=("Segoe UI", 11, "bold"),
                         bg=CARD, fg=ACCENT2).pack(side=tk.LEFT, padx=12)

    # ── Camera Pipeline ──────────────────────────────────────
    def toggle_camera(self):
        if not self.camera_running:
            self.camera_running = True
            self._cam_btn.config(text="⏹  Stop Camera", bg=RED)
            self._status_var.set("● Live")
            t = threading.Thread(target=self._run_pipeline, daemon=True)
            t.start()
        else:
            self.camera_running = False
            self._cam_btn.config(text="▶  Start Camera", bg=GREEN)
            self._status_var.set("● Ready")
            self._video_lbl.config(image="", text="Camera Off")

    def _run_pipeline(self):
        import time as _time
        detector          = HandDetector()
        feature_extractor = FeatureExtractor()
        classifier        = Classifier()
        assembler         = WordAssembler()
        self._assembler   = assembler   # allow _clear_all to reset it
        prev_label        = ""
        consecutive       = 0
        last_hand_ts      = 0.0        # for hand-switch grace period
        _last_display_ts  = 0.0        # for 30fps display cap
        _DISPLAY_INTERVAL = 1.0 / 30   # cap UI redraws at 30fps

        try:
            with Camera() as cam:
                while self.camera_running:
                    ret, frame = cam.get_frame()
                    if not ret:
                        continue

                    proc, _  = detector.find_hands(frame.copy(), draw=True)
                    lm_list  = detector.get_landmarks(proc, hand_no=0)
                    hand_present = bool(lm_list)
                    label, conf  = "", 0.0

                    if hand_present:
                        features    = feature_extractor.extract_features(lm_list)
                        label, conf = classifier.predict(features)

                        # Confidence gate: reject predictions the model is unsure about.
                        # This is the primary fix – without this, the model's bias
                        # toward high-sample classes (I, H, E) dominates.
                        if conf < config.MIN_PREDICTION_CONFIDENCE:
                            label = ""

                        # Stability gate – must see same letter N frames in a row
                        if label == prev_label:
                            consecutive += 1
                        else:
                            consecutive = 0
                            prev_label  = label

                        # Threshold: 5 stable frames ≈ 0.5s at 10 inferences/sec.
                        # Fast enough for fluid signing, stable enough to avoid noise.
                        if consecutive > 4 and label not in ("", "Unknown"):
                            assembler.push_letter(label)
                            consecutive = 0   # reset so same letter needs re-stabilising

                    # Tick assembler every frame ─ updates word/sentence state
                    # Grace period: don't report hand absent until 0.5 s after
                    # last detection — absorbs brief drops during hand switches.
                    now = _time.time()
                    if hand_present:
                        last_hand_ts = now
                    grace_present = hand_present or (now - last_hand_ts) < config.HAND_LOST_GRACE_SEC
                    state = assembler.tick(grace_present)

                    # ── Update the four display tiers ────────────────────
                    buf = state["word_buffer"]
                    self._buf_var.set("  ".join(buf) if buf else "–")

                    # Live word prediction – update every frame for instant feedback
                    pred = assembler.live_prediction()
                    self._pred_var.set(f"→ {pred}" if pred and pred != "".join(buf) else "")

                    if state["last_word"]:
                        self._word_var.set(state["last_word"])

                    if state["sentence"]:
                        self._sent_var.set(state["sentence"])

                    # ── Word completed event ──────────────────────────────
                    cw = state["completed_word"]
                    if cw:
                        self._log_append(f"✔ Word: {cw}")
                        self.tts.speak_async(cw)          # speak the whole word
                        self.tracker.log_translation(cw, 1.0, "camera")

                    # ── Sentence completed event ──────────────────────────
                    cs = state["completed_sentence"]
                    if cs:
                        self._log_append(f"📢 Sentence: {cs}")
                        self.tracker.log_translation(cs, 1.0, "camera")
                        # Only re-speak if sentence has 2+ words (1-word sentences
                        # were already spoken at word-boundary time above)
                        if len(cs.split()) > 1:
                            self.tts.speak_async(cs)
                        # Reset display for next sentence
                        self._buf_var.set("–")
                        self._pred_var.set("")
                        self._word_var.set("–")
                        self._sent_var.set("")

                    # ── Render frame at capped 30fps to reduce CPU load ────────
                    now_ts = _time.time()
                    if now_ts - _last_display_ts >= _DISPLAY_INTERVAL:
                        self._show_frame(self._video_lbl, proc)
                        _last_display_ts = now_ts

        except Exception as e:
            self._status_var.set(f"● Error: {e}")
        finally:
            self._assembler = None

    # ── Learning Mode Pipeline ───────────────────────────────
    def toggle_learn(self):
        if not self.learn_running:
            self.learn_running = True
            self._learn_btn.config(text="⏹  Stop Practice", bg=RED)
            sign = self.current_sign.get()
            tip  = self.learning.get_lesson(sign)
            self._tip_var.set(f"💡 {tip}")
            t = threading.Thread(target=self._run_learn, daemon=True)
            t.start()
        else:
            self.learn_running = False
            self._learn_btn.config(text="▶  Start Practice", bg=GREEN)
            self._learn_lbl.config(image="", text="Learning Mode Paused")
            self._fb_var.set("")

    def _run_learn(self):
        try:
            with Camera() as cam:
                while self.learn_running:
                    ret, frame = cam.get_frame()
                    if not ret:
                        continue
                    sign = self.current_sign.get()
                    result_img, feedback = self.learning.evaluate_sign(frame, sign)
                    self._fb_var.set(feedback)
                    self._show_frame(self._learn_lbl, result_img)
        except Exception as e:
            self._fb_var.set(f"Error: {e}")

    # ── Image Upload ─────────────────────────────────────────
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        label, conf, result_img = self.image_translator.translate(path)
        self._img_result.set(label)
        self.tracker.log_translation(label, conf, "image")
        if label not in ("No hand detected", "Error: Image not found"):
            self.tts.speak_async(label)
        self._show_frame(self._img_lbl, result_img)

    # ── Utility ──────────────────────────────────────────────
    def _show_frame(self, label_widget, cv_img):
        try:
            rgb  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            w    = label_widget.winfo_width()  or 640
            h    = label_widget.winfo_height() or 480
            pil  = Image.fromarray(rgb).resize((w, h), Image.BILINEAR)
            tk_img = ImageTk.PhotoImage(image=pil)
            label_widget.config(image=tk_img, text="")
            label_widget.image = tk_img  # keep reference
        except Exception:
            pass

# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = BridgeSignApp(root)

    def _on_close():
        app.camera_running = False
        app.learn_running  = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
