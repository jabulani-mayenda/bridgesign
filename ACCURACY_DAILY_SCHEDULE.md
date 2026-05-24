# BridgeSign Daily Accuracy Schedule

Use this before presentation practice so the live hand-sign module improves every day instead of drifting.

## Daily 30 Minute Loop

1. Run the report:
   `python tools/daily_accuracy_report.py`

2. Spend 10 minutes on the weakest static hand signs:
   use the report's `data_collector.py` command and record varied distance, lighting, and hand angle.

3. Spend 10 minutes on the weakest live motion signs:
   use the report's `gesture_collector.py` command, especially `J`, `Z`, `STATIC`, and high-priority words like `HELP`, `STOP`, `WATER`, and `THANK_YOU`.

4. Retrain after collection:
   `python model_trainer.py`
   `python gesture_trainer.py`
   `python lstm_gesture_trainer.py`

5. Run the audit:
   `python _system_audit.py`

6. Presentation smoke test:
   in Motion, type `I want to do all`, `HELLO THANK_YOU`, and `HELP WATER`.
   in Live, test the lowest-count letters plus `J`, `Z`, `HELP`, and `STOP`.

## Target

Static letters should stay at 220+ samples each. Motion and word signs should move toward 180+ total samples, with at least 40 fresh samples added for weak labels on collection days.
