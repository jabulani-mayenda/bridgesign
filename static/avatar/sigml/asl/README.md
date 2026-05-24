# BridgeSign ASL SiGML Draft Dictionary

This folder contains local SiGML files loaded by the CWASA avatar.

Status:
- The dictionary covers A-Z plus the critical/emergency word signs from `word_signs.py`.
- Each label has its own SiGML file; the player no longer aliases several signs to one placeholder.
- These are still reviewable drafts. For production-level ASL precision, have an ASL signer or HamNoSys/SiGML author tune handshape, contact, movement, and non-manual markers.

File naming:
- One uppercase label per file, for example `HELLO.sigml`.
- Labels should match app sign labels such as `THANK_YOU`, `SORRY`, `PLEASE`.

Regenerate:
- Edit `tools/build_asl_sigml_dictionary.py`.
- Run `python tools/build_asl_sigml_dictionary.py`.

How to improve a sign:
1. Open the sign in the CWASA player.
2. Compare against a trusted ASL reference video.
3. Adjust handshape, orientation, location, movement, and non-manual markers.
4. Keep the XML well formed.
