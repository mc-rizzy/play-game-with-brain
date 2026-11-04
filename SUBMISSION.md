# Hackathon submission — play-game-with-brain

Team name: Team 01f-w8c9spo

## Which track are you submitting to?

Indie Developer Track.

## Which problem statement did you build towards?

**Push the Frontier**: give a machine a new capability — control a video
game purely with thought. An 8-electrode EEG headset streams brainwaves
through a dual-layer meta-pipeline (rolling PSD features -> spatial GNN
-> per-player few-shot adapter, coached by Meta's Muse models) that steers
a live Pygame world alongside the keyboard. No controllers needed.

## Project description

A brain-computer interface for gaming. The BrainAccess Mini headset (8
channels @ 250 Hz) streams raw EEG; instead of feeding raw samples to a
network (high latency, unstable training), a Feature Extraction Layer
computes rolling 0.5 s power-spectral-density features across
theta/alpha/beta/gamma bands, a Spatial GNN Layer message-passes over
electrode nodes with spatial + functional-connectivity edges, and a
meta-model head personalizes to each new brain in a 15-second in-game
Calibration Wizard (press C). Live inference writes commands to the game
at 4 Hz; keyboard keeps working simultaneously. Includes training
(local + Modal cloud), a realtime EEG dashboard, replay mode for
hardware-free demos, and an 82-test suite.

## Describe your use of Meta products to build the submitted project

- **Meta Model API + Muse Spark (`muse-spark-1.3`)** — called live from
  `gnn/muse.py` (OpenAI-compatible `https://api.meta.ai/v1`) in two
  pipeline roles: (1) calibration coach that reads each player's fresh
  EEG summary and returns personal coaching + tuned live gates, and
  (2) optional intent referee (`--muse-fallback`) that disambiguates
  gray-zone frames from prediction history + game context. Auth via the
  hackathon `META_TOKEN` in `.env`. Every call is timed, printed to the
  console, and archived in `calibration_report.json` (`muse.calls`), so
  the demo video and repo both show the Meta model calls.
- **Muse Code** — the agentic coding assistant used to build this
  project (pipeline, wizard, tests, docs).

## Why are Meta's models important to your solution?

The GNN learns signal patterns but cannot explain them or reason about a
new player. Every brain is wired differently: fixed confidence gates that
work for one judge misfire for the next, and hand-written tuning rules
cannot read a separability report. Muse closes that loop — it turns 15
seconds of signal statistics into human coaching ("hold each imagery
steady through the full cue") plus machine-usable gate thresholds, and it
referees ambiguous moments with context no classifier has. Without Meta's
models the game plays; with them it understands *this* player.

## Why was Meta's API and Muse models integral for this project?

Personalization is the product: a brain-game that only works for the
person it trained on is a party trick. The Calibration Wizard + Muse
coach is what makes a stranger (a judge) playable in 15 seconds — the
GNN adapts its head while Muse adapts the experience. The referee role
further converts would-be dropped inputs into correct moves. Both sit on
the critical path of every demo run, not in a side panel.

## What was the biggest obstacle your team hit while building this project, and how did Meta's models and API solve them?

Two obstacles. First, **cross-brain generalization**: one player's motor
imagery looks like another player's noise, and we could not ship a
labeled dataset per judge. The few-shot adapter + Muse coach solved it —
the adapter fits the head in seconds while Muse reads the adaptation
report and tunes gates/coaching per brain. Second, **silent uncertainty**:
the old pipeline dropped every sub-threshold frame, which feels like
lag. The Muse referee recovers the recoverable ones using context,
instead of frozen heuristics.

## Compared with other AI models, how did Meta's models outperform on this project?

Framed honestly: we built on Muse because the Meta Model API is
drop-in OpenAI-compatible (base URL + key + model id, stdlib HTTP —
no new SDK in our pipeline), the context window swallows full
calibration summaries plus prediction history in one call, and the model
reliably returns the strict JSON our game loop parses (coaching +
thresholds + verdict), verified live during development. That combination
— compatibility, context, structured output — is what the realtime loop
needed.

## For this project, when choosing an AI model or coding tool, what matters most?

Latency profile honesty (we learned Muse Spark spends tokens reasoning,
so we budget for it and keep it off the 4 Hz path), structured-output
reliability (the game parses every reply), and drop-in API compatibility
so the BCI loop stays dependency-light.

## What makes it hardest for your team to switch to a different AI model or coding tool?

The pipeline now depends on Muse's specific behavior: coaching quality
grounded in EEG statistics, threshold tuning that respects our gate
ranges, and the audited call trail judges inspect. Re-validating that
closed loop — prompts, JSON contract, offline fallbacks, demo timing —
against another model would cost more than the integration itself.

## Provide feedback from your experience using Modal

Modal training (`gnn/modal_train.py`) worked as designed: featurization
stays local, only tiny feature tensors ship to the worker, and the same
`model.pt` comes back. Auth via `.env` tokens plus `modal setup` fallback
was smooth; the main friction was credit-watching on a student budget,
which is why local training remains a first-class identical path.

## Judge demo flow (3 minutes)

1. `python -m game.main` — show keyboard play (0:00-0:30).
2. Press **C** — 15 s Calibration Wizard on the judge's own head
   (`--source live`; `--source demo` without hardware), watch the Muse
   coaching + tuned gates land live (0:30-1:30).
3. `python -m gnn.live --adapter ... --muse-fallback` — play with thought;
   show `calibration_report.json` `muse.calls` as the Meta-model audit
   trail (1:30-3:00).
