# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
Use a structured key-value format on two separate lines:

  Label: <label>
  Reasoning: <one or two sentences>

Tradeoff analysis:
- JSON is machine-readable but LLMs often wrap it in markdown fences
  (```json ... ```) requiring extra stripping and json.loads() error handling.
- Label-on-its-own-line is simplest to parse but the LLM sometimes adds a
  preamble sentence before the label, breaking a naive "first line" approach.
- "Label: X / Reasoning: Y" (chosen): easy to scan all lines for one that
  starts with "label:", split on ":", take the value. Robust to extra lines
  before or after. Normalize with .lower().strip() to handle capitalization
  variance (e.g., "Label: Interview" → "interview") and strip markdown
  punctuation like ** or `` to handle bold/code formatting.
```

---

**Edge cases to handle in the prompt:**

```
1. labeled_examples is empty: the prompt still sends the task instruction and
   the new episode — the LLM falls back to zero-shot classification using the
   label definitions in the instruction block. Include the full label definitions
   in the instruction so the model has enough context even without examples.

2. Very short description (< ~20 characters): include it as-is. The LLM handles
   short inputs and will lean on the label definitions. No special handling needed.

3. Description contains special characters or newlines: Python f-strings handle
   these naturally; no escaping required for a plain-text prompt.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
Scan every line of the response for lines that start with "label:" and
"reasoning:" (case-insensitive). Split each matched line on the first ":"
and take the right-hand side. Strip whitespace and markdown punctuation
(*, _, `) from the extracted label before validation.

  label = "unknown"
  reasoning = response_text.strip()
  for line in response_text.strip().split("\n"):
      clean = line.strip()
      if clean.lower().startswith("label:"):
          label = clean.split(":", 1)[1].strip().lower().strip("*_` ")
      elif clean.lower().startswith("reasoning:"):
          reasoning = clean.split(":", 1)[1].strip()

Scanning all lines (not just the first) makes parsing robust to preamble
sentences the LLM sometimes inserts before the structured output.
```

---

**Step 4 — Validate the label:**

```
After parsing, check whether the extracted label is in VALID_LABELS
(["interview", "solo", "panel", "narrative"]). If it is not — because the
LLM returned an unrecognized string, left the field blank, or the label line
was never found — set label to "unknown". Never pass an unvalidated string
back to the caller.

  if label not in VALID_LABELS:
      label = "unknown"
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the entire function body in a try/except Exception block. Possible
failures include: network timeout, Groq API rate-limit error, empty response,
or an unparseable response that slips past the parsing logic.

On any exception, return a safe fallback dict so the evaluation loop
(which calls this function 20 times) can continue without crashing:

  except Exception as e:
      return {"label": "unknown", "reasoning": f"Error during classification: {e}"}

This guarantees that one bad API call never aborts the entire evaluation run.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: [title]
Raw response text: [paste it here]
```

**How did you parse the label out of the response?**

```
[describe the string operations — strip, split, lower, etc.]
```

**Did any episodes return `"unknown"`? If so, why?**

```
[yes / no — if yes, what did the raw response look like?]
```

**One thing about the output format that surprised you:**

```
[your answer here]
```
