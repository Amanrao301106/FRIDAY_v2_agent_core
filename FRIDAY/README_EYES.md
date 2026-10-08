# FRIDAY Eyes Upgrade

This is an additive upgrade for the existing FRIDAY v2 agent.

## What it adds

1. `vision/capture.py` — safe screenshot capture.
2. `vision/client.py` — Ollama vision-model adapter.
3. `vision/analyzer.py` — converts a screenshot into structured targets.
4. `vision/controller.py` — confidence-gated mouse/keyboard actions.
5. `browser/controller.py` — safe URL/search helpers.
6. `eyes.py` — high-level `FridayEyes.observe()` API.
7. Tests for the new components.

## Important model requirement

Your existing `llama3` text model should not be assumed to understand images.
Configure a vision-capable Ollama model such as `llava` (or another vision model
you have installed).

Example:

```python
from vision.client import OllamaVisionClient
from vision.analyzer import ScreenAnalyzer
from eyes import FridayEyes

vision = OllamaVisionClient(model="llava")
eyes = FridayEyes(ScreenAnalyzer(vision))

obs = eyes.observe("Find the search box")
print(obs.summary)
for target in obs.targets:
    print(target.label, target.x, target.y, target.confidence)
```

## Install

Add to the existing FRIDAY environment:

```powershell
pip install pillow pytest
```

PyAutoGUI and ollama are already part of the existing FRIDAY dependency set.

If you use Ollama locally, install/pull a vision model separately according to
the model's Ollama instructions.

## Safety design

- Screenshot is observation only.
- Clicks are blocked when confidence is below the configured threshold.
- `confirm=True` intentionally refuses to perform the action; the FRIDAY
  orchestrator should obtain human approval before calling a sensitive action.
- Browser helpers allow only `http://` and `https://`.
- No arbitrary shell execution is added.

## Next integration point

Wire `FridayEyes.observe()` into the existing agent/orchestrator as a new
tool/action:

`observe_screen -> choose_target -> click/type -> observe_screen -> verify`

Sensitive actions remain confirmation-gated.
