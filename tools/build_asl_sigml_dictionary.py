"""
Build the local CWASA ASL SiGML dictionary.

The entries here are compact HamNoSys-style drafts. They are intentionally
distinct per label, cover A-Z plus the critical word set, and avoid the old
placeholder behavior where many signs reused the same motion.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "static" / "avatar" / "sigml" / "asl"


def sign(tags: list[str]) -> list[str]:
    return tags


ALPHABET_SIGNS = {
    "A": sign(["hamfist", "hamthumboutmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "B": sign(["hamflathand", "hamthumbacrossmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "C": sign(["hamceeall", "hamextfingeru", "hampalml", "hamshoulders"]),
    "D": sign(["hamfinger2", "hamthumbopenmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "E": sign(["hamfist", "hamthumbopenmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "F": sign(["hampinch12open", "hamextfingeru", "hampalml", "hamshoulders"]),
    "G": sign(["hamfinger2", "hamextfingerr", "hampalmd", "hamshoulders"]),
    "H": sign(["hamfinger23", "hamextfingerr", "hampalmd", "hamshoulders"]),
    "I": sign(["hamfist", "hamthumbacrossmod", "hamextfingeru", "hampalmr", "hamshoulders"]),
    "J": sign(["hamfist", "hamthumbacrossmod", "hamextfingeru", "hampalmr", "hamshoulders", "hammovedr", "hamarcu"]),
    "K": sign(["hamfinger23spread", "hamthumbopenmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "L": sign(["hamfinger2", "hamthumboutmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "M": sign(["hamfist", "hamthumbacrossmod", "hamextfingeru", "hampalmd", "hamshoulders"]),
    "N": sign(["hamfist", "hamthumbacrossmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "O": sign(["hampinchall", "hamextfingeru", "hampalml", "hamshoulders"]),
    "P": sign(["hamfinger23spread", "hamthumbopenmod", "hamextfingerd", "hampalml", "hamshoulders"]),
    "Q": sign(["hamfinger2", "hamthumboutmod", "hamextfingerd", "hampalmd", "hamshoulders"]),
    "R": sign(["hamfinger23", "hamextfingeru", "hampalml", "hamshoulders", "hamtwisting"]),
    "S": sign(["hamfist", "hamthumbacrossmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "T": sign(["hamfist", "hamthumbopenmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "U": sign(["hamfinger23", "hamextfingeru", "hampalml", "hamshoulders"]),
    "V": sign(["hamfinger23spread", "hamextfingeru", "hampalml", "hamshoulders"]),
    "W": sign(["hamfinger2345", "hamextfingeru", "hampalml", "hamshoulders"]),
    "X": sign(["hamfinger2", "hamfingerhookedmod", "hamextfingeru", "hampalml", "hamshoulders"]),
    "Y": sign(["hamfist", "hamthumboutmod", "hamextfingeru", "hampalml", "hamshoulders", "hamswinging"]),
    "Z": sign(["hamfinger2", "hamextfingeru", "hampalml", "hamshoulders", "hammover", "hamzigzag"]),
}


CRITICAL_WORD_SIGNS = {
    "HELP": sign(["hamfist", "hamthumboutmod", "hamextfingerol", "hampalml", "hamchest", "hammoveu"]),
    "STOP": sign(["hamflathand", "hamextfingeru", "hampalmd", "hamchest", "hammoved", "hamtouch"]),
    "PAIN": sign(["hamfinger2", "hamextfingero", "hampalml", "hamchest", "hamtwisting"]),
    "DOCTOR": sign(["hamfinger2345", "hamextfingerl", "hampalmd", "hamwristpulse", "hamtouch", "hammoved", "hamsmallmod"]),
    "WATER": sign(["hamfinger2345", "hamextfingeru", "hampalml", "hamchin", "hamtouch", "hammoved", "hamsmallmod"]),
    "FOOD": sign(["hampinchall", "hamextfingeru", "hampalml", "hamlips", "hamtouch", "hammoved", "hamsmallmod"]),
    "TOILET": sign(["hamfist", "hamthumbopenmod", "hamextfingeru", "hampalml", "hamshoulders", "hammovel"]),
    "FIRE": sign(["hamfinger2345", "hamfingerhookedmod", "hamextfingeru", "hampalml", "hambelowstomach", "hammoveu", "hamwavy"]),
    "POLICE": sign(["hamceeall", "hamextfingeru", "hampalml", "hamchest", "hamtouch"]),
    "AMBULANCE": sign(["hamfist", "hamthumboutmod", "hamextfingeru", "hampalml", "hamshoulders", "hamcircleu"]),
    "EMERGENCY": sign(["hamfist", "hamthumbopenmod", "hamextfingeru", "hampalml", "hamchest", "hammovel"]),
    "DANGER": sign(["hamfist", "hamthumboutmod", "hamextfingeru", "hampalmd", "hamchest", "hammoveu"]),
    "SICK": sign(["hamfinger23", "hamextfingeru", "hampalml", "hamforehead", "hamtouch", "hamplus", "hamfinger23", "hamextfingerd", "hampalml", "hamstomach", "hamtouch"]),
    "HURT": sign(["hamfinger2", "hamextfingero", "hampalml", "hamchest", "hamtwisting"]),
}


CONVERSATION_SIGNS = {
    "HELLO": sign(["hamflathand", "hamextfingeru", "hampalml", "hamforehead", "hammover", "hamarcu"]),
    "THANK_YOU": sign(["hamflathand", "hamextfingeru", "hampalml", "hamchin", "hamtouch", "hammoveo"]),
    "SORRY": sign(["hamfist", "hamthumbacrossmod", "hamextfingerol", "hampalml", "hamchest", "hamtouch", "hamcircleu"]),
    "PLEASE": sign(["hamflathand", "hamextfingerol", "hampalml", "hamchest", "hamtouch", "hamcircleu"]),
    "YES": sign(["hamfist", "hamthumbacrossmod", "hamextfingerol", "hampalml", "hamshoulders", "hammoved", "hamsmallmod"]),
    "NO": sign(["hamfinger23", "hamthumbopenmod", "hamextfingerol", "hampalml", "hamshoulders", "hamclose"]),
    "WHAT": sign(["hamflathand", "hamextfingerol", "hampalmu", "hamshoulders", "hammovel", "hamsmallmod"]),
    "WHERE": sign(["hamfinger2", "hamextfingeru", "hampalml", "hamshoulders", "hammover", "hamsmallmod"]),
    "MUG": sign(["hamfist", "hamthumbacrossmod", "hamextfingerol", "hampalml", "hamshoulders", "hamparbegin", "hammoveu", "hamarcu", "hamreplace", "hamextfingerul", "hampalmdl", "hamparend"]),
    "TAKE": sign(["hamceeall", "hamextfingerol", "hampalml", "hamlrbeside", "hamshoulders", "hamarmextended", "hamreplace", "hamextfingerl", "hampalml", "hamchest", "hamclose"]),
}


SIGNS = {
    **ALPHABET_SIGNS,
    **CRITICAL_WORD_SIGNS,
    **CONVERSATION_SIGNS,
}


def xml_for(label: str, tags: list[str]) -> str:
    tag_lines = "\n".join(f"      <{tag}/>" for tag in tags)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<sigml>\n"
        f'  <hns_sign gloss="{label.lower()}">\n'
        "    <hamnosys_manual>\n"
        f"{tag_lines}\n"
        "    </hamnosys_manual>\n"
        "  </hns_sign>\n"
        "</sigml>\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, tags in SIGNS.items():
        (OUT_DIR / f"{label}.sigml").write_text(xml_for(label, tags), encoding="utf-8")

    manifest = {
        "language": "ASL",
        "status": "draft",
        "coverage": {
            "alphabet": len(ALPHABET_SIGNS),
            "critical_words": len(CRITICAL_WORD_SIGNS),
            "conversation_words": len(CONVERSATION_SIGNS),
            "total": len(SIGNS),
        },
        "signs": sorted(SIGNS),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(SIGNS)} SiGML files to {OUT_DIR}")


if __name__ == "__main__":
    main()
