# L06 Coaching Session — Facilitator Run-Sheet

**Title:** Make It Better, Make It Trustworthy — Improving the HDB Price Model
**Duration:** 2 hours (incl. 1 hour hands-on)
**Audience:** Adult learners, non-technical, who completed L01–L05 (built & deployed a 3-feature HDB price predictor in Streamlit)
**Pedagogy:** Kolb's experiential learning cycle

---

## Learning outcomes
By the end, learners can:
1. Explain what **MAE** is and read it as "the model is on average off by S$X".
2. Use a **train/test split** and say why testing on unseen data is the honest way to score a model.
3. **Add features** to a model and observe the error change.
4. **Compare two models** and pick the better one by *test* error.
5. Recognise **overfitting** from a train-vs-test gap.

## Before the session (facilitator)
- Run `python model.py` once yourself to confirm the dataset downloads and note the actual MAE numbers (they refresh with the live data).
- Ensure learners' environments from L05 still work (`pip install -r requirements.txt`).
- Open `explainer.html` in a browser, ready to project.
- Have the trap-flat examples ready (see Hook).

## Materials in this folder
| File | Used in | Purpose |
|---|---|---|
| `explainer.html` | Concept segment | Project & click through MAE / features / overfitting |
| `notebooks/L06_practice.ipynb` | Hands-on hour | Learners fill in blanks, measure & compare |
| `model.py`, `app.py` | Wrap-up / homework | Upgraded training + Streamlit app |
| `requirements.txt` | Setup | Dependencies |

---

## Timeline (Kolb cycle mapped)

### 1. Concrete Experience — "Catch it being wrong" · 0:00–0:12 (12 min)
- Open the **L05 Streamlit app**. Ask the room to predict prices for two **trap flats**:
  - A *large flat in a cheap, far town* (e.g. 130 sqm in Woodlands).
  - A *small flat in a prime town* (e.g. 45 sqm in Bishan / Central).
- The model gives confidently wrong-looking numbers because it can't see *flat type* or *town*.
- **Prompt:** "Would you stake your commission on this number?" Let the discomfort sit.

### 2. Reflective Observation — "Why did it miss?" · 0:12–0:25 (13 min)
- Facilitate, don't lecture. Surface answers on a whiteboard:
  - *What does the model actually know?* (only size, lease year, floor)
  - *What does it never get told?* (type, location, condition…)
  - *How would we even know how wrong it is?* → motivates a score.
- Land the key question: **"How do we measure 'good', and how do we make it better?"**

### 3. Abstract Conceptualization — Three ideas, with the explainer · 0:25–0:50 (25 min)
Project `explainer.html`. One tab per idea, ~7–8 min each. Keep it interactive — have a learner drive the slider.
- **Tab 1 — MAE:** "average dollar gap." Slide tightness; watch the red error bars and the dollar figure move. Analogy: a weather forecast that's "on average 3°C off."
- **Tab 2 — Adding features:** toggle *flat type* and *town* on; watch the error bar shrink. Message: *more relevant information usually helps.*
- **Tab 3 — Overfitting & model choice:** slide complexity; show the U-shaped test-error curve vs the always-falling training curve. Analogy: the student who memorised past papers. Message: *pick the lowest **test** error, not the lowest training error.*
- Briefly name the two models they'll try: **Linear Regression** (straight-line) vs **Random Forest** (captures combinations).

### 4. Active Experimentation — Hands-on practice · 0:50–1:50 (60 min)
Learners open `notebooks/L06_practice.ipynb`. Circulate and coach.
- **0:50–1:00 — Part A:** rebuild the 3-feature model and read its MAE. Everyone writes their baseline number.
- **1:00–1:20 — Part B (fill the blank):** add `flat_type` + `town`, compare MAE to baseline. Checkpoint: "Did your error drop? Shout out your before/after."
- **1:20–1:45 — Part C (fill the blank):** swap in a Random Forest; compare test errors; inspect the train-vs-test gap. Checkpoint: "Which model would you ship, and why?"
- **1:45–1:50 — Challenge** for fast finishers (extra feature / more trees / find the worst prediction).

*Coaching cues:* if someone is stuck on a blank, point them to the **▸ Hint**, not the answer. Common trip-ups: forgetting the dataset needs internet; mixing up train vs test error direction.

### 5. Debrief & Consolidation · 1:50–2:00 (10 min)
- Two or three learners share before/after MAE and their model choice.
- Recap the wrap-up table (MAE, train/test, features, model choice, overfitting).
- **Homework / next step:** plug the better model into the upgraded `app.py` (`python model.py` then `streamlit run app.py`) and redeploy — now with flat type and town inputs.

---

## Facilitator answer key (the blanks)
- **Part B blank:** `columns=category`
- **Part C blank:** `RandomForestRegressor`

## Likely questions & quick answers
- *"Why not just add every column?"* — Irrelevant or leaky columns can add noise or cheat (e.g. block number). Add features that plausibly drive price.
- *"Is Random Forest always better?"* — No. It often wins here, but it's slower and can overfit. Always check the test error.
- *"Why does my MAE differ from my neighbour's?"* — The live dataset updates; small differences are normal. The *direction* of change is the lesson.
