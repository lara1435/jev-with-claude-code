#!/usr/bin/env python3
"""UserPromptSubmit hook: let Jev answer simple judgments before the LLM.

One Jev request asks a routing Choice plus speculative answers of each kind:
  - noul:   yes/no questions
  - choice: which emotion a text expresses
  - score:  how negative/positive a text's sentiment is
If Jev confidently routes to one of those and its answer is confident, the
prompt is blocked and Jev's answer is shown in yellow. Otherwise (or on any
error) the prompt passes through to Claude unchanged.
"""
import json
import os
import sys
import urllib.request

API_URL = "https://api.typesafe.ai/v1/systemone"
ROUTE_THRESHOLD = 0.7    # min probability of the chosen route
NOUL_MARGIN = 0.15       # yes/no answer must be <= this or >= 1 - this
CHOICE_THRESHOLD = 0.6   # min confidence for the emotion choice
SCORE_THRESHOLD = 0.5    # min confidence for the sentiment score
YELLOW, RESET = "\033[33m", "\033[0m"

EMOTIONS = {
    "joy": "Happiness, delight, excitement, gratitude",
    "love": "Affection, warmth, care, admiration toward someone",
    "sadness": "Grief, disappointment, loneliness, regret",
    "anger": "Hostility, rage, hatred, irritation",
    "fear": "Worry, anxiety, dread, nervousness",
    "surprise": "Astonishment or shock, positive or negative",
    "disgust": "Revulsion, contempt, strong distaste",
    "neutral": "No clear emotion; factual or flat",
}

SENTIMENT_LEVELS = [
    "Very negative: hateful, furious, devastated, or deeply hostile",
    "Negative: unhappy, critical, annoyed, or disappointed",
    "Neutral: factual, mixed, or no clear feeling",
    "Positive: pleased, friendly, satisfied, or hopeful",
    "Very positive: overjoyed, loving, ecstatic, or deeply grateful",
]


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


def ask_jev(prompt, api_key):
    subject = "the text the user in `prompt` is asking about (or `prompt` itself if no text is quoted)"
    body = {
        "state": {"prompt": prompt},
        "model": "jev-latest",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "What kind of answer does the request in `prompt` need?",
                "criteria": {
                    "yes_no": "A single yes or no fully answers it (e.g. 'is this text happy?', 'is Paris in France?')",
                    "emotion": "It asks which emotion or feeling a piece of text expresses",
                    "sentiment_rating": "It asks how positive or negative a piece of text is, or to rate its sentiment",
                    "needs_llm": (
                        "Anything else: writing or editing code, reading or changing files, running "
                        "commands, explanations, summaries, open-ended questions, or multi-step work"
                    ),
                },
            },
            "yes_no": {
                "type": "noul",
                "instructions": "Assume `prompt` asks a yes/no question. What is the correct answer?",
                "criteria": {"true": "Yes", "false": "No"},
            },
            "emotion": {
                "type": "choice",
                "instructions": f"Which emotion does {subject} mainly express?",
                "criteria": EMOTIONS,
            },
            "sentiment": {
                "type": "score",
                "instructions": f"How negative or positive is the sentiment of {subject}?",
                "criteria": SENTIMENT_LEVELS,
            },
        },
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def format_answer(route, answers):
    """Return Jev's answer text for the chosen route, or None if not confident."""
    if route == "yes_no":
        p_yes = answers["yes_no"]["noul"]
        if NOUL_MARGIN < p_yes < 1 - NOUL_MARGIN:
            return None
        return f"{'YES' if p_yes >= 0.5 else 'NO'} (noul p_yes={p_yes:.2f})"

    if route == "emotion":
        emotion = answers["emotion"]
        if emotion["confidence"] < CHOICE_THRESHOLD:
            return None
        top = sorted(emotion["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        dist = ", ".join(f"{k}={v:.2f}" for k, v in top)
        return f"{emotion['choice'].upper()} (choice confidence={emotion['confidence']:.2f}; {dist})"

    if route == "sentiment_rating":
        sentiment = answers["sentiment"]
        if sentiment["confidence"] < SCORE_THRESHOLD:
            return None
        level = sentiment["legend"][str(round(sentiment["score"]))].split(":")[0]
        return (
            f"{level.upper()} (score {sentiment['score']:.2f} on 0-4, "
            f"confidence={sentiment['confidence']:.2f})"
        )

    return None


def main():
    try:
        prompt = json.load(sys.stdin).get("prompt", "").strip()
        api_key = load_api_key()
        if not prompt or prompt.startswith("/") or not api_key:
            return

        result = ask_jev(prompt, api_key)
        answers = result["answers"]
        route = answers["route"]["choice"]
        route_p = answers["route"]["probabilities"][route]
        if route == "needs_llm" or route_p < ROUTE_THRESHOLD:
            return  # needs more: fall through to the LLM

        text = format_answer(route, answers)
        if text is None:
            return  # Jev unsure: fall through to the LLM

        message = f"{YELLOW}Jev ({result.get('model', 'jev')}) [{route}]: {text}{RESET}"
        print(json.dumps({"decision": "block", "reason": message}))
    except Exception:
        return  # fail open: never block the prompt because Jev is unavailable


if __name__ == "__main__":
    main()
