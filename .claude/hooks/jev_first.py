#!/usr/bin/env python3
"""UserPromptSubmit hook: let Jev answer simple judgments before the LLM.

1. A routing Choice decides what kind of answer the prompt needs.
2. Code builds one question from the prompt itself and sends only that:
     - yes_no:      noul
     - pick_option: choice over options listed in the prompt ("a, b or c")
     - rate_scale:  score over the scale named in the prompt ("from 0 to 5");
                    choice over the numbers when the scale has > 10 levels
If the route and the answer are confident, the prompt is blocked and Jev's raw
response for that one question is shown in yellow. Otherwise (no options found,
low confidence, or any error) the prompt passes through to Claude unchanged.
"""
import json
import os
import re
import sys
import urllib.request

API_URL = "https://api.typesafe.ai/v1/systemone"
ROUTE_THRESHOLD = 0.7    # min probability of the chosen route
NOUL_MARGIN = 0.15       # yes/no answer must be <= this or >= 1 - this
CHOICE_THRESHOLD = 0.9   # min confidence for a choice answer
SCORE_THRESHOLD = 0.5    # min confidence for a score answer
NEAR_SCALE_THRESHOLD = 0.7  # min probability within +/-1 of the pick on a >10-level scale
MAX_SCORE_LEVELS = 10    # API limit for Score; larger scales use Choice
MAX_CHOICE_OPTIONS = 255
YELLOW, RESET = "\033[33m", "\033[0m"

ROUTE_QUESTION = {
    "type": "choice",
    "instructions": "What kind of answer does the request in `prompt` need?",
    "criteria": {
        "yes_no": "A single yes or no fully answers it",
        "pick_option": (
            "It asks to pick one of several alternatives that are written out in the prompt "
            "(e.g. 'is it X, Y or Z?', 'A or B?', 'choose between A and B')"
        ),
        "rate_scale": "It asks for a rating or number on a numeric scale stated in the prompt (e.g. 'from 0 to 10', 'out of 5')",
        "needs_llm": (
            "Anything else: writing or editing code, reading or changing files, running "
            "commands, explanations, summaries, open-ended questions, or multi-step work"
        ),
    },
}

QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"|“[^”]*”|‘[^’]*’")
SCALE_PATTERNS = [
    re.compile(r"(-?\d+)\s*(?:to|-|–|through)\s*(-?\d+)", re.I),
    re.compile(r"between\s+(-?\d+)\s+and\s+(-?\d+)", re.I),
]
OUT_OF = re.compile(r"(?:out of|/)\s*(\d+)", re.I)
TRAILING_CLAUSE = re.compile(
    r"\s+(?:for|to|in|on|at|with|when|because|if|since|during|given|as)\b.*$", re.I
)


def extract_options(prompt):
    """Find the alternatives the prompt lists, e.g. 'happy, sad or angry'."""
    text = QUOTED.sub(" ", prompt).strip().rstrip("?.!").strip()

    # Explicit list after a colon: "choose one: red, green, blue"
    if ":" in text:
        tail = text.rsplit(":", 1)[1]
        items = [s.strip() for s in re.split(r",|\bor\b|/|\|", tail) if s.strip()]
        if len(items) >= 2:
            return dedupe(items)

    # Slash- or pipe-separated: "positive/negative/neutral"
    for token in text.split():
        parts = [p for p in re.split(r"[/|]", token) if p]
        if len(parts) >= 2 and not any(p.isdigit() for p in parts):
            return dedupe(parts)

    # List after a cue word: "between A and B", "among A, B, C", "as A, B or C"
    # (also tolerates misspellings like "beween")
    m = re.search(r"\b(?:betw?e+n|beween|among(?:st)?|from|as|into|options?|choices?)\s+(.+)$", text, re.I)
    if m:
        items = re.split(r",|\s+(?:and|or)\s+|/|\|", m.group(1))
        items[-1] = TRAILING_CLAUSE.sub("", items[-1])
        options = dedupe(items)
        if options:
            return options

    # "..., A, B or C" / "A or B"
    if re.search(r"\bor\b", text):
        left, right = re.split(r"\s+or\s+", text, maxsplit=1, flags=re.I) if text.lower().count(" or ") == 1 \
            else text.rsplit(" or ", 1)
        right = TRAILING_CLAUSE.sub("", right).strip(" ,")
        chunks = [c.strip() for c in left.split(",") if c.strip()]
        if not chunks or not right:
            return []
        later = chunks[1:] + [right]
        width = max(len(o.split()) for o in later)
        first = " ".join(chunks[0].split()[-width:])
        return dedupe([first] + later)

    return []


def dedupe(items):
    seen, out = set(), []
    for item in (i.strip(" ,.;") for i in items):
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out if len(out) >= 2 else []


def extract_scale(prompt):
    """Find the numeric scale the prompt names; return (low, high) or None."""
    text = QUOTED.sub(" ", prompt)
    for pattern in SCALE_PATTERNS:
        m = pattern.search(text)
        if m:
            lo, hi = sorted((int(m.group(1)), int(m.group(2))))
            if lo < hi:
                return lo, hi
    m = OUT_OF.search(text)
    if m and int(m.group(1)) > 1:
        return 1, int(m.group(1))
    return None


def build_question(route, prompt):
    """Build the single answer question for this route from the prompt, or None."""
    if route == "yes_no":
        return {
            "type": "noul",
            "instructions": "Answer the yes/no question in `prompt`.",
            "criteria": {"true": "Yes", "false": "No"},
        }

    if route == "pick_option":
        options = extract_options(prompt)
        if not 2 <= len(options) <= MAX_CHOICE_OPTIONS:
            return None
        return {
            "type": "choice",
            "instructions": "Answer the request in `prompt` by picking the one option that fits best.",
            "criteria": {o: None for o in options},
        }

    if route == "rate_scale":
        scale = extract_scale(prompt)
        if scale is None:
            return None
        lo, hi = scale
        numbers = list(range(lo, hi + 1))
        if len(numbers) > MAX_CHOICE_OPTIONS:
            return None
        label = {lo: f"{lo}: the lowest end of the scale described in `prompt`",
                 hi: f"{hi}: the highest end of the scale described in `prompt`"}
        instructions = f"Answer the rating request in `prompt` on its {lo}-to-{hi} scale."
        if len(numbers) <= MAX_SCORE_LEVELS:
            return {
                "type": "score",
                "instructions": instructions,
                "criteria": [label.get(n, str(n)) for n in numbers],
            }
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": {str(n): label.get(n) for n in numbers},
        }

    return None


def load_api_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()), ".env")
    try:
        with open(env_path) as f:
            for line in f:
                name, _, value = line.strip().partition("=")
                if name == "TYPESAFE_API_KEY":
                    return value.strip().strip("\"'")
    except OSError:
        pass
    return None


def ask_jev(prompt, questions, api_key):
    body = {"state": {"prompt": prompt}, "model": "jev-latest", "questions": questions}
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def is_confident(answer):
    """Whether Jev's single answer is confident enough to skip the LLM."""
    if answer["type"] == "noul":
        return not NOUL_MARGIN < answer["noul"] < 1 - NOUL_MARGIN
    if answer["type"] == "choice":
        if all(k.lstrip("-").isdigit() for k in answer["probabilities"]):
            # Ordered number scale: mass near the pick counts, not just the pick itself.
            pick = int(answer["choice"])
            near = sum(v for k, v in answer["probabilities"].items() if abs(int(k) - pick) <= 1)
            return near >= NEAR_SCALE_THRESHOLD
        return answer["confidence"] >= CHOICE_THRESHOLD
    if answer["type"] == "score":
        return answer["confidence"] >= SCORE_THRESHOLD
    return False


def main():
    try:
        prompt = json.load(sys.stdin).get("prompt", "").strip()
        api_key = load_api_key()
        if not prompt or prompt.startswith("/") or not api_key:
            return

        route_answer = ask_jev(prompt, {"route": ROUTE_QUESTION}, api_key)["answers"]["route"]
        route = route_answer["choice"]
        if route == "needs_llm" or route_answer["probabilities"][route] < ROUTE_THRESHOLD:
            return  # needs more: fall through to the LLM

        question = build_question(route, prompt)
        if question is None:
            return  # no options/scale found in the prompt: fall through to the LLM

        result = ask_jev(prompt, {route: question}, api_key)
        if not is_confident(result["answers"][route]):
            return  # Jev unsure: fall through to the LLM

        message = f"{YELLOW}{json.dumps(result, indent=2)}{RESET}"
        print(json.dumps({"decision": "block", "reason": message}))
    except Exception:
        return  # fail open: never block the prompt because Jev is unavailable


if __name__ == "__main__":
    main()
