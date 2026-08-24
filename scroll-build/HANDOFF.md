# Evidra scroll landing — asset handoff

Manual asset path. You render the images/videos; I validate + wire them into the site.

- **Style:** clay diorama, **neon-lime on near-black graphite** (matches the new app
  color scheme — #080C0E background, #76FF03 lime accent — baked into every prompt)
- **Camera:** walkthrough (continuous forward glide) — start-frame only, so it works
  with all your video tools (Kling, Luma, and Gemini as fallback)
- **Scenes:** 6 · **Assets:** 6 stills + 6 forward "leg" clips (no connectors)
- **Image tool:** ChatGPT
- **Video tools:** Kling for clips 1–3, Luma for clips 4–6, Gemini only as a fallback
  for the last clip if credits run out. Switch models in a contiguous block (Kling then
  Luma), never alternate, to keep only one model-boundary seam.

---

## ▶ PHASE 1 — render the 6 still images (do this now)

For each row: open the prompt file, copy its **entire** contents, paste into ChatGPT's
image generator, download the result, and save it into `scroll-build/stills/` with the
exact filename in the "Save as" column.

**Rules that matter (already written into every prompt, don't change them):**
- Wide **landscape** image (ask ChatGPT for landscape / 3:2 if it offers a choice).
- Plain solid **#0a0a0b** (near-black) background, nothing behind the object.
- **No text/letters/numbers/logos** in the image.
- Keep the subject **centred** with a little space above it.
- Use **ChatGPT for all 6** (one tool = one consistent look). Don't mix in other tools.

| # | Scene | Prompt file | Save as | Status |
|---|-------|-------------|---------|--------|
| 1 | Raw data | `prompts/still_1_rawdata.txt` | `stills/still_1_rawdata.png` | ⬜ pending |
| 2 | Semantic profiling | `prompts/still_2_profiling.txt` | `stills/still_2_profiling.png` | ⬜ pending |
| 3 | Agent loop (LangGraph) | `prompts/still_3_agentloop.txt` | `stills/still_3_agentloop.png` | ⬜ pending |
| 4 | Cross-validation (5-fold) | `prompts/still_4_crossval.txt` | `stills/still_4_crossval.png` | ⬜ pending |
| 5 | Decision gate | `prompts/still_5_decision.txt` | `stills/still_5_decision.png` | ⬜ pending |
| 6 | Champion + report (hero) | `prompts/still_6_champion.txt` | `stills/still_6_champion.png` | ⬜ pending |

When all 6 are saved, tell me. I'll check they're the right size/aspect and that the 6
read as one cohesive world, then hand you **Phase 2** (animate each still into a forward
camera clip in Gemini).

---

## Copy per scene (for later — the words that appear over each scene)

1. **Raw data** — eyebrow: *From messy CSV* · title: "It starts with raw data." · body: "Upload any tabular dataset — Evidra takes it from there."
2. **Profiling** — eyebrow: *Understand* · title: "It reads your data." · body: "A semantic profile of every column, no config required."
3. **Agent loop** — eyebrow: *Reason* · title: "A LangGraph agent, on a loop." · body: "It forms hypotheses and tests them across a research budget."
4. **Cross-validation** — eyebrow: *Validate* · title: "Five-fold, no leakage." · body: "Every idea is proven with rigorous cross-validation."
5. **Decision gate** — eyebrow: *Decide* · title: "Only real gains pass." · body: "A 0.5% threshold keeps the noise out."
6. **Champion** — eyebrow: *Ship* · title: "An evidence-backed pipeline." · body: "Launch the research workspace." · **CTA button → /overview**

_(You can tweak any of this wording later; it's easy to change in code.)_

---

Phase 1 stills: ✅ all 6 rendered and validated (1536×1024, 3:2, cohesive).

---

## ▶ PHASE 2 — render the 6 motion clips (do this now)

Each clip **starts from its matching still** and adds a slow camera move. They're
independent — **render them in any order, in parallel**, and the scroll engine crossfades
between them. Split across your tools: **Kling → clips 1–3, Luma → clips 4–6.**

**Settings for every clip:**
- **Start / first frame = the matching still** in `scroll-build/stills/` (upload it as the
  image, then paste the prompt text).
- **Aspect: 16:9 landscape.** Duration **~5s** (Kling default is fine; Luma ~5s).
- **No audio**, highest quality the tool offers. No end-frame needed.
- Save each result into `scroll-build/clips/` with the exact name below.

| # | Start image (still) | Prompt file | Tool | Save clip as | Status |
|---|---------------------|-------------|------|--------------|--------|
| 1 | `stills/still_1_rawdata.png` | `prompts/clip_1_rawdata.txt` | Kling | `clips/clip_1_rawdata.mp4` | ✅ encoded (watermark cropped) |
| 2 | `stills/still_2_profiling.png` | `prompts/clip_2_profiling.txt` | **Gemini** | `clips/clip_2_profiling.mp4` | ⛔ RE-RENDER — current file holds the champion scene, not profiling |
| 3 | `stills/still_3_agentloop.png` | `prompts/clip_3_agentloop.txt` | Gemini | `clips/clip_3_agentloop.mp4` | ✅ encoded |
| 4 | `stills/still_4_crossval.png` | `prompts/clip_4_crossval.txt` | Gemini | `clips/clip_4_crossval.mp4` | ✅ encoded |
| 5 | `stills/still_5_decision.png` | `prompts/clip_5_decision.txt` | Gemini | `clips/clip_5_decision.mp4` | ✅ encoded |
| 6 | `stills/still_6_champion.png` | `prompts/clip_6_champion.txt` | Gemini | `clips/clip_6_champion.mp4` | ✅ encoded |

**Only clip 2 (profiling) is outstanding.** Render it in Gemini with the updated
`clip_2_profiling.txt` (start image = `still_2_profiling.png`), overwrite
`clips/clip_2_profiling.mp4`, and tell me — I'll encode it to
`frontend/public/world/profiling.mp4` and wire up the scroll page. The other 5 are already
encoded into `frontend/public/world/`.

### Gemini (Veo) rendering notes
- **Use image-to-video:** upload the matching **still as the input / first frame**, then
  paste the clip prompt. That keeps the clip locked to the scene you already approved.
- If your Gemini plan only does text-to-video (no image input), it will still generate
  from the prompt but the scene may drift from the still — tell me and I'll adjust.
- **Aspect 16:9**, keep it landscape.
- **Audio:** Veo adds sound automatically — ignore it, I strip audio during encoding.
- If a clip comes out too fast/jittery, re-run once; Veo is non-deterministic and the
  "slow, graceful, eases to a gentle stop" wording usually settles it on the 2nd try.

---

## Phases after this (preview — nothing to do yet)

- **Phase 3** — I install/encode the clips for smooth scroll-scrubbing (needs `ffmpeg`)
  and wire the scroll engine into the landing page.
- **Phase 4** — I polish the rest of the app (dashboard, upload, overview) with modern
  Framer Motion animations. No assets needed from you.
