"""PII: personal and patient data in slides and frames (`detection pii`, `detection-skill/pii@1`, ADR-006).

OCR text lines (`detection text`) are checked against patterns for names, patient ids, dates of birth, phone numbers
and addresses, stated per locale; faces (`detection faces`, optional) are listed too, for clinical photos on slides.
Every hit becomes a redact-ready rectangle (the line's or face's box, padded, even pixels; in a video with its time
span) with the category, the rule and the reason. Every region is a proposal (`decision: CONFIRM`): nothing is
removed here, and nothing is dropped silently; a person confirms each one before ffmpeg-skill `redact` runs.

- **Locales.** `intl` (the default) holds international and English patterns: ISO 8601 / day-month-year / month-day-year
  / month-name dates after a birth keyword, E.164 phone numbers, English keywords (Patient ID, MRN, DOB, Name,
  Address, Tel). Regional profiles are opt-in: `en-US` (NANP phones, MM/DD/YYYY, ZIP codes, SSN), `en-GB` (NHS
  number, UK phones and postcodes), `ja-JP` (患者ID / カルテ番号, 生年月日 with 和暦 or 西暦, 電話, 〒 and 住所, 氏名 / 様).
  The patterns of every profile are listed by `detection contract --json` and in docs/detections.md.
- **Keywords first.** A date is a date of birth only next to a birth keyword; a number is a patient id only next to an
  id keyword; a name only after a name keyword or an honorific. A bare name or a bare date on a slide is not found:
  this is a pattern check, not a named-entity recogniser, and it says so in every document.
- **Boxes.** The region is the whole OCR line (a line box over-covers the hit, which is safer for redaction than a
  guessed sub-box). The matched text is not copied into the document: `excerpt` keeps the keyword and masks digits
  and letters of the value.
"""
from __future__ import annotations

import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

from . import __version__
from .detections import ValidationReport, _check_run_fields, _num, check_source, scan_leakage, utc_now, validate_detections
from .engines.boxes import iou
from .errors import DetectionError
from .tracks import redact_rect

PII_SCHEMA = "detection-skill/pii@1"
TOOL = "detection/pii"
CATEGORIES = ("name", "patient_id", "date_of_birth", "phone", "address", "face")
DEFAULT_LOCALES = ("intl",)
DEFAULT_PAD = 0.1
DECISION = "CONFIRM"
REQUIRED = ("schema", "source", "inputs", "method", "parameters", "regions", "redact", "counts", "skill", "skill_version", "tool",
            "created_at", "processing_seconds", "warnings")
NOTE = ("pattern check, not a named-entity recogniser: a value is found only next to its keyword (or, for a phone number, "
        "in a phone format); a bare name or date is not found; OCR errors can hide a hit; every region is a proposal to "
        "confirm, nothing is removed")
REGION_ID_RE = re.compile(r"^pii_\d{3,}$")

_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_INTL = (r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}\s+" + _MONTHS + r"\.?,?\s+\d{2,4}|"
              + _MONTHS + r"\.?\s+\d{1,2},?\s+\d{2,4})")
_ID_VALUE = r"[A-Z]{0,4}[-\s]?\d[\d\-\s]{3,15}\d"


def _r(p: str) -> Pattern[str]:
    return re.compile(p, re.IGNORECASE)


# (rule id, category, compiled pattern, what it means). The value group is named "v" where there is one.
RULES: Dict[str, List[Tuple[str, str, Pattern[str], str]]] = {
    "intl": [
        ("intl.patient_id", "patient_id",
         _r(r"\b(?:patient\s*(?:id|no\.?|number|#)|pt\.?\s*id|mrn|medical\s+record\s*(?:no\.?|number|#)?|hospital\s*(?:no\.?|number)|"
            r"case\s*(?:no\.?|number|#))\s*[:#.]?\s*(?P<v>" + _ID_VALUE + r")"),
         "an id after a patient id keyword (Patient ID, MRN, medical record, hospital number, case no.)"),
        ("intl.date_of_birth", "date_of_birth",
         _r(r"\b(?:d\.?o\.?b\.?|date\s+of\s+birth|birth\s*date|born(?:\s+on)?)\s*[:.]?\s*(?P<v>" + _DATE_INTL + r")"),
         "a date after a birth keyword (DOB, date of birth, born)"),
        ("intl.phone_e164", "phone", re.compile(r"(?P<v>\+\d{1,3}[\s\-.]?(?:\(?\d{1,4}\)?[\s\-.]?){2,4}\d{2,4})"),
         "an international phone number (+country code, E.164 style)"),
        ("intl.phone_keyword", "phone",
         _r(r"\b(?:tel|phone|mobile|cell|fax)\.?\s*[:.]?\s*(?P<v>\+?\(?\d[\d\s\-().]{5,18}\d)"),
         "a number after a phone keyword (Tel, Phone, Mobile, Fax)"),
        ("intl.address", "address", _r(r"\b(?:address|addr\.)\s*[:.]\s*(?P<v>\S.{3,})"),
         "text after an address keyword"),
        ("intl.name", "name",
         _r(r"\b(?:patient(?:'s)?\s+name|patient|name|pt)\s*[:]\s*(?P<v>[^\W\d_][^\W\d_'\-.]*(?:[\s,]+[^\W\d_][^\W\d_'\-.]*){0,3})"),
         "words after a name keyword (Patient:, Name:, Patient name:)"),
        ("intl.name_title", "name",
         re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx)\.?\s+(?P<v>[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2})"),
         "a capitalised name after Mr / Mrs / Ms / Miss / Mx"),
    ],
    "en-US": [
        ("en-US.ssn", "patient_id", _r(r"\b(?:ssn|social\s+security(?:\s+(?:no\.?|number))?)\s*[:#.]?\s*(?P<v>\d{3}-?\d{2}-?\d{4})\b"),
         "a US social security number after its keyword"),
        ("en-US.phone", "phone", re.compile(r"(?<![\d.])(?P<v>(?:\+?1[\s\-.])?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4})(?![\d])"),
         "a North American phone number ((555) 123-4567, 555-123-4567)"),
        ("en-US.address", "address",
         _r(r"(?P<v>\b\d{1,6}\s+(?:[A-Z][\w.]*\s+){1,4}(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|"
            r"way|place|pl)\b\.?(?:.*\b\d{5}(?:-\d{4})?\b)?)"),
         "a street address (number, name, Street / Ave / Rd ...), optionally with a ZIP code"),
    ],
    "en-GB": [
        ("en-GB.nhs_number", "patient_id", _r(r"\bnhs\s*(?:no\.?|number|#)?\s*[:.]?\s*(?P<v>\d{3}[\s\-]?\d{3}[\s\-]?\d{4})\b"),
         "an NHS number after its keyword"),
        ("en-GB.phone", "phone", re.compile(r"(?<![\d+])(?P<v>0\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4})(?!\d)"),
         "a UK phone number (0 and 9-10 more digits)"),
        ("en-GB.postcode", "address", re.compile(r"\b(?P<v>[A-Z]{1,2}\d[A-Z\d]?\s+\d[A-Z]{2})\b"),
         "a UK postcode"),
    ],
    "ja-JP": [
        ("ja-JP.patient_id", "patient_id",
         re.compile(r"(?:患者\s*(?:ID|ＩＤ|番号|No\.?)|カルテ\s*(?:番号|No\.?)|診察券\s*番号|ID)\s*[:：]?\s*(?P<v>[0-9０-９][0-9０-９\-－]{3,15})"),
         "患者ID / 患者番号 / カルテ番号 / 診察券番号 の後の番号"),
        ("ja-JP.date_of_birth", "date_of_birth",
         re.compile(r"(?:生年月日|誕生日)\s*[:：]?\s*(?P<v>(?:(?:明治|大正|昭和|平成|令和|[MTSHR])\s*[0-9０-９元]{1,2}|[0-9０-９]{4})\s*[年./／]"
                    r"\s*[0-9０-９]{1,2}\s*[月./／]\s*[0-9０-９]{1,2}\s*日?)"),
         "生年月日 / 誕生日 の後の日付（和暦・西暦）"),
        ("ja-JP.phone", "phone", re.compile(r"(?<![0-9０-９])(?P<v>0[0-9]{1,4}[\-－(（][0-9]{1,4}[\-－)）][0-9]{3,4})(?![0-9])"),
         "日本の電話番号（0 から始まる市外局番つき）"),
        ("ja-JP.postal", "address", re.compile(r"(?P<v>〒\s*[0-9０-９]{3}[\-－][0-9０-９]{4}.*)"), "〒 郵便番号（とその後の住所）"),
        ("ja-JP.address", "address", re.compile(r"(?:住所|所在地)\s*[:：]?\s*(?P<v>\S.{3,})"), "住所 の後の文字列"),
        ("ja-JP.name", "name", re.compile(r"(?:氏名|患者名|名前)\s*[:：]?\s*(?P<v>[^\s:：0-9０-９]{1,10}(?:[\s　][^\s:：0-9０-９]{1,10})?)"),
         "氏名 / 患者名 の後の文字列"),
        ("ja-JP.honorific", "name", re.compile(r"(?P<v>[一-龯々ァ-ヴー]{1,6}[\s　]?[一-龯々ァ-ヴーぁ-ん]{0,6})[\s　]?(?:様|殿)(?![ぁ-ん])"),
         "様 / 殿 の前の氏名"),
    ],
}
LOCALES = tuple(RULES)


def parse_locales(text: Optional[Sequence[str]]) -> List[str]:
    if not text:
        return list(DEFAULT_LOCALES)
    out: List[str] = []
    for v in text:
        for part in str(v).split(","):
            part = part.strip()
            if not part:
                continue
            if part not in RULES:
                raise DetectionError("INVALID_ARGUMENT", f"--locale {part!r}: one of {', '.join(LOCALES)}")
            if part not in out:
                out.append(part)
    return out or list(DEFAULT_LOCALES)


def mask(text: str) -> str:
    """Digits become 0-free '#', letters '*'; separators and spacing stay, so the shape is kept, the value is not."""
    return "".join("#" if ch.isdigit() else ("*" if ch.isalpha() else ch) for ch in text)


def find(text: str, locales: Sequence[str]) -> List[Dict[str, Any]]:
    """Every rule hit in one OCR line: [{rule, category, start, end, excerpt, means}]; overlapping hits of one category
    are kept once (the first rule wins)."""
    hits: List[Dict[str, Any]] = []
    for loc in locales:
        for rule, cat, pat, means in RULES[loc]:
            for m in pat.finditer(text):
                vs, ve = (m.start("v"), m.end("v")) if "v" in pat.groupindex and m.group("v") is not None else (m.start(), m.end())
                if any(h["category"] == cat and not (ve <= h["start"] or vs >= h["end"]) for h in hits):
                    continue
                excerpt = (text[m.start():vs] + mask(text[vs:ve])).strip()
                hits.append({"rule": rule, "category": cat, "start": vs, "end": ve, "excerpt": excerpt, "means": means})
    hits.sort(key=lambda h: (h["start"], CATEGORIES.index(h["category"])))
    return hits


def _ref(doc: Dict[str, Any], file: Optional[str], fingerprint: Optional[str]) -> Dict[str, Any]:
    return {"file": os.path.basename(file) if file else None, "fingerprint": fingerprint, "tool": doc.get("tool"),
            "kind": doc.get("kind"), "frames": doc["counts"]["frames"], "detections": doc["counts"]["detections"],
            "lang": (doc.get("parameters") or {}).get("lang"), "level": (doc.get("parameters") or {}).get("level")}


def make_pii(text: Dict[str, Any], *, faces: Optional[Dict[str, Any]] = None, locales: Optional[Sequence[str]] = None,
             pad: Optional[float] = None, files: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None) -> Dict[str, Any]:
    t0 = time.time()
    files = files or {}
    rep = validate_detections(text)
    if not rep.ok:
        raise DetectionError("INVALID_DETECTIONS", "text: not a valid detections document", rep.errors[:10])
    if text["kind"] != "text":
        raise DetectionError("INVALID_ARGUMENT", "the input must be text detections (detection text)")
    if faces is not None:
        rep = validate_detections(faces)
        if not rep.ok:
            raise DetectionError("INVALID_DETECTIONS", "--faces: not a valid detections document", rep.errors[:10])
        if faces["kind"] != "face":
            raise DetectionError("INVALID_ARGUMENT", "--faces must be face detections (detection faces)")
        if faces["source"]["fingerprint"] != text["source"]["fingerprint"]:
            raise DetectionError("INVALID_ARGUMENT", "--faces was measured on another file (source fingerprints differ)")
    locs = parse_locales(locales)
    pad = DEFAULT_PAD if pad is None else float(pad)
    if not math.isfinite(pad) or not 0 <= pad <= 1:
        raise DetectionError("INVALID_ARGUMENT", "--pad must be in [0, 1]")
    width, height = text["frame_size"]["width"], text["frame_size"]["height"]
    warnings: List[str] = []
    if (text.get("parameters") or {}).get("level") == "word":
        warnings.append("word-level OCR: a keyword and its value in different words are not matched; use --level line")
    regions: List[Dict[str, Any]] = []
    for fr in text["frames"]:
        for d in fr["detections"]:
            hits = find(d["label"] or "", locs)
            for cat in CATEGORIES:
                hs = [h for h in hits if h["category"] == cat]
                if not hs:
                    continue
                regions.append({"t": fr["t"], "category": cat, "box": list(d["box"]), "rules": [h["rule"] for h in hs],
                                "excerpt": " | ".join(h["excerpt"] for h in hs),
                                "reason": f"{cat.replace('_', ' ')} in OCR text: " + "; ".join(h["means"] for h in hs),
                                "source": "text", "score": d["score"]})
    if faces is not None:
        for fr in faces["frames"]:
            for d in fr["detections"]:
                regions.append({"t": fr["t"], "category": "face", "box": list(d["box"]), "rules": ["face"], "excerpt": None,
                                "reason": "a face (a clinical photo or a person in the frame): confirm whether it may be shown",
                                "source": "faces", "score": d["score"]})
    regions.sort(key=lambda r: ((r["t"] if r["t"] is not None else -1.0), r["box"][1], r["box"][0], CATEGORIES.index(r["category"])))
    for i, r in enumerate(regions):
        r["id"] = f"pii_{i + 1:03d}"
        r["decision"] = DECISION
        r["redact"] = redact_rect([r["box"]], pad, width, height)
    step = _step(text)
    redact = _spans(regions, step, float(text["source"]["duration"]) if _num(text["source"].get("duration")) else None)
    by_cat = {c: sum(1 for r in regions if r["category"] == c) for c in CATEGORIES}
    if not regions:
        warnings.append("no region found: this is a pattern check (keywords and formats), so review the slides as well")
    inputs = {"text": _ref(text, *files.get("text", (None, None))),
              "faces": _ref(faces, *files.get("faces", (None, None))) if faces is not None else None}
    return {
        "schema": PII_SCHEMA,
        "source": dict(text["source"]),
        "inputs": inputs,
        "method": {"name": "patterns", "heuristic": True, "identity": False, "note": NOTE},
        "parameters": {"locales": locs, "pad": pad, "categories": list(CATEGORIES),
                       "rules": [r[0] for loc in locs for r in RULES[loc]] + (["face"] if faces is not None else [])},
        "regions": regions,
        "redact": redact,
        "counts": {"regions": len(regions), "redact": len(redact), "by_category": by_cat, "confirm": len(regions)},
        "skill": "detection-skill",
        "skill_version": __version__,
        "tool": TOOL,
        "created_at": utc_now(),
        "processing_seconds": round(time.time() - t0, 3),
        "warnings": warnings,
    }


def _step(doc: Dict[str, Any]) -> Optional[float]:
    sm = doc.get("sampling") or {}
    if sm.get("method") == "every" and _num(sm.get("every")):
        return float(sm["every"])
    ts = sm.get("times") or []
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    return min(gaps) if gaps else None


def _spans(regions: List[Dict[str, Any]], step: Optional[float], duration: Optional[float]) -> List[Dict[str, Any]]:
    """Regions of one category whose boxes overlap (IoU >= 0.5) at consecutive sample times become one redact span:
    [first t, last t + one step] (the image itself, start/end null, for an image)."""
    spans: List[Dict[str, Any]] = []
    for r in regions:
        t = r["t"]
        for s in spans:
            if s["category"] == r["category"] and iou(s["_box"], r["box"]) >= 0.5 and (
                    t is None or (step is not None and 0 < t - s["_last"] <= step * 1.5 + 1e-6)):
                s["regions"].append(r["id"])
                s["_last"] = t if t is not None else s["_last"]
                s["_box"] = r["box"]
                break
        else:
            spans.append({"category": r["category"], "regions": [r["id"]], "_last": t, "_first": t, "_box": r["box"]})
    out = []
    for s in spans:
        first = next(r for r in regions if r["id"] == s["regions"][0])
        xs = [r["redact"] for r in regions if r["id"] in s["regions"]]
        x1, y1 = min(x["x"] for x in xs), min(x["y"] for x in xs)
        x2, y2 = max(x["x"] + x["width"] for x in xs), max(x["y"] + x["height"] for x in xs)
        if s["_first"] is None:
            start = end = None
        else:
            start = round(s["_first"], 3)
            end = round(s["_last"] + (step or 0.0), 3)
            if duration is not None:
                end = min(end, round(duration, 3))
        out.append({"category": s["category"], "regions": s["regions"], "start": start, "end": end, "x": x1, "y": y1,
                    "width": x2 - x1, "height": y2 - y1, "decision": DECISION, "reason": first["reason"]})
    return out


def validate_pii(doc: Any) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("pii document must be a JSON object")
        return rep
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != PII_SCHEMA:
        rep.errors.append(f"schema must be {PII_SCHEMA}")
    _check_run_fields(doc, rep, TOOL)
    check_source(doc["source"], rep)
    src = doc["source"] if isinstance(doc["source"], dict) else {}
    W, H = src.get("width"), src.get("height")
    m = doc["method"]
    if not isinstance(m, dict) or m.get("heuristic") is not True or m.get("identity") is not False:
        rep.errors.append("method must declare heuristic: true and identity: false")
    par = doc["parameters"]
    if not isinstance(par, dict) or not isinstance(par.get("locales"), list) or not par["locales"] \
            or any(loc not in RULES for loc in par["locales"]):
        rep.errors.append(f"parameters.locales must list locales from {LOCALES}")
    regions = doc["regions"]
    if not isinstance(regions, list):
        rep.errors.append("regions must be a list")
        return rep
    ids = set()
    for i, r in enumerate(regions):
        k = f"regions[{i}]"
        if not isinstance(r, dict) or any(f not in r for f in ("id", "t", "category", "box", "rules", "reason", "decision", "redact")):
            rep.errors.append(f"{k}: needs id, t, category, box, rules, reason, decision, redact")
            continue
        if not isinstance(r["id"], str) or not REGION_ID_RE.match(r["id"]) or r["id"] in ids:
            rep.errors.append(f"{k}: id must be a unique pii_nnn")
        ids.add(r["id"])
        if r["category"] not in CATEGORIES:
            rep.errors.append(f"{k}: category must be one of {CATEGORIES}")
        if r["decision"] != DECISION:
            rep.errors.append(f"{k}: decision must be {DECISION} (every region is a proposal; nothing is removed silently)")
        if not isinstance(r["reason"], str) or not r["reason"]:
            rep.errors.append(f"{k}: needs a reason")
        if not isinstance(r["rules"], list) or not r["rules"]:
            rep.errors.append(f"{k}: needs the rule(s) that found it")
        _check_rect(r["redact"], W, H, f"{k}.redact", rep)
    redact = doc["redact"]
    if not isinstance(redact, list):
        rep.errors.append("redact must be a list")
        redact = []
    covered = set()
    for i, s in enumerate(redact):
        k = f"redact[{i}]"
        if not isinstance(s, dict) or s.get("decision") != DECISION or not isinstance(s.get("regions"), list) or not s["regions"]:
            rep.errors.append(f"{k}: needs regions and decision {DECISION}")
            continue
        covered.update(s["regions"])
        _check_rect(s, W, H, k, rep)
        if (s.get("start") is None) != (s.get("end") is None) or (s.get("start") is not None and (
                not _num(s["start"]) or not _num(s["end"]) or s["end"] < s["start"])):
            rep.errors.append(f"{k}: start <= end, or both null for an image")
    if covered != ids:
        rep.errors.append("every region must be in exactly the redact spans (none dropped)")
    c = doc["counts"]
    if not isinstance(c, dict) or c.get("regions") != len(regions) or c.get("redact") != len(redact) \
            or c.get("confirm") != len(regions):
        rep.errors.append("counts do not match")
    scan_leakage(doc, "", rep)
    return rep


def _check_rect(r: Any, W: Any, H: Any, k: str, rep: ValidationReport) -> None:
    ok = isinstance(r, dict) and all(isinstance(r.get(f), int) and not isinstance(r.get(f), bool) for f in ("x", "y", "width", "height"))
    if not ok or r["width"] < 2 or r["height"] < 2 or r["width"] % 2 or r["height"] % 2 or r["x"] < 0 or r["y"] < 0 \
            or (isinstance(W, int) and r["x"] + r["width"] > W) or (isinstance(H, int) and r["y"] + r["height"] > H):
        rep.errors.append(f"{k}: needs even integer width/height >= 2 inside the frame")


__all__ = ["make_pii", "validate_pii", "find", "mask", "parse_locales", "PII_SCHEMA", "LOCALES", "RULES"]
