import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    """
    parts = []

    # Section 1: task instruction
    parts.append(
        "You are classifying podcast episodes by their format.\n"
        "Classify the episode into exactly one of these four labels:\n"
        "- interview: a host speaks with one or more guests; structured as questions and answers\n"
        "- solo: a single host speaking alone with no guests, sharing their own thoughts or experience\n"
        "- panel: multiple speakers with roughly equal participation, discussing a topic together\n"
        "- narrative: a story assembled from external sources (reporting, archives, interviews) with a clear story arc\n\n"
        "Respond using EXACTLY this two-line format and nothing else before it:\n"
        "Label: <label>\n"
        "Reasoning: <one or two sentences explaining your choice>"
    )

    # Section 2: labeled examples
    if labeled_examples:
        parts.append("Here are labeled examples to learn from:\n")
        for ep in labeled_examples:
            parts.append(
                f"---\n"
                f"Title: {ep['title']}\n"
                f"Description: {ep['description']}\n"
                f"Label: {ep['label']}"
            )

    # Section 3: new episode to classify
    parts.append(
        f"---\n"
        f"Now classify this new episode:\n"
        f"Description: {description}\n\n"
        f"Remember: respond with exactly:\n"
        f"Label: <one of: interview, solo, panel, narrative>\n"
        f"Reasoning: <brief explanation>"
    )

    return "\n\n".join(parts)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    try:
        # Step 1: build the prompt
        prompt = build_few_shot_prompt(labeled_examples, description)

        # Step 2: send to the LLM
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        response_text = response.choices[0].message.content

        # Step 3: parse — scan all lines for "Label:" and "Reasoning:"
        label = "unknown"
        reasoning = response_text.strip()
        for line in response_text.strip().split("\n"):
            clean = line.strip()
            if clean.lower().startswith("label:"):
                label = clean.split(":", 1)[1].strip().lower().strip("*_` ")
            elif clean.lower().startswith("reasoning:"):
                reasoning = clean.split(":", 1)[1].strip()

        # Step 4: validate — reject anything not in VALID_LABELS
        if label not in VALID_LABELS:
            label = "unknown"

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error during classification: {e}"}
