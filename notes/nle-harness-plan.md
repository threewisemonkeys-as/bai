# Running the Claude Code agent on NetHack

Plan for a fourth sibling harness, `cc_nle/`, after `cc_autumn/` (AutumnBench),
`cc_humanrl/` (the six prior-ablated MonsterKong games) and `cc_craftax/` (Craftax).
Target: the [NetHack Learning Environment](https://github.com/NetHack-LE/nle), Küttler
et al. NeurIPS 2020 ([paper](https://arxiv.org/abs/2006.13760)), now maintained by the
NetHack-LE organisation — `nle==1.3.0`, which is NetHack 3.6.6 compiled as a gym
environment.

**Scope: the game and its interface as shipped.** No renamed or remapped keys, no
natural-language action wrapper, no auto-dismissed menus, no withheld reward, and every
observation the agent receives is one the package itself produces. What changes is only
who is playing it — a coding-agent session instead of a policy network or a symbolic
bot — and the number we get out is in the benchmark's own units, next to the
benchmark's own baselines.

This matters more here than it did for Craftax, because there is already an
LLM-on-NetHack number in this repository: BALROG runs agents on `NetHackChallenge-v0`
through `nle-language-wrapper`, which renames the keys into English (`"far northeast"`,
`"apply"`), translates the screen into prose, and can auto-skip `--More--`. That is a
different environment with a friendlier interface and its own published numbers. Here
the agent gets the terminal.

Everything under "Findings" was measured against `nle==1.3.0` installed into
`cc_nle/nle-code/.env-venv` on this box, not read off the README. Where a number is
quoted, the probe that printed it is named.

---

## 0. What the benchmark actually is

One game — NetHack 3.6.6, complete and unmodified — exposed through nine registered
environments that differ in what they ask of it.

| | **NetHackChallenge-v0** | **NetHackScore-v0** and the task suite |
|---|---|---|
| actions | 121 — the full keyboard | 23 — compass, stairs, wait, a few commands |
| menus / `--More--` | **not skipped** | NLE steps past them for you |
| character | rolled at random each episode | rolled at random each episode |
| seeding | **refused** (anti-TAS, F5) | allowed |
| step cap | 10⁶, plus a no-progress abort at 10⁴ | 10⁶ |
| reward | in-game score delta | score delta − 0.01 per frozen step |
| what it is for | the NeurIPS 2021 competition | the NLE paper's RL baselines |

The dungeon is ~50 levels deep with branches, a quest, four demon lords and an
endgame. There is no goal short of ascension, which 1.46% of human games on
nethack.alt.org reach and no artificial agent has ever reached. The task is: score as
much as you can before you die.

**The baselines we are joining.** Two families, and they are far apart.

| NetHack Challenge 2021 (median score over 4,096 games) | | NLE-paper RL (NetHackScore, ~10⁹ steps) | |
|---|---|---|---|
| AutoAscend (symbolic) | **5,336.5** | CDGPT5 / IMPALA-class | ~700–1,000 |
| Students of Stone (symbolic) | 2nd | | |
| RAPH (best "neural", hybrid) | ~1/3 of AutoAscend | | |

*"The top symbolic agent beat the top neural agent by a factor of almost 3 in the
median score"*, and in over half a million evaluation games **no agent ascended**
(Hambro et al., *Insights From the NeurIPS 2021 NetHack Challenge*). One in twenty of
the winner's Valkyrie games reached dungeon level 10 and experience level 10.

And the LLM number, from BALROG (Paglieri et al., ICLR 2025), on their language-wrapped
Challenge: **o1-preview 1.57% progression, Claude-3.5-Sonnet 1.16%**, most models below
0.6%, and the best single run in the whole benchmark reached **dungeon level 3,
experience level 4**. That is the number a coding-agent harness is actually competing
with, and it is very low.

---

## Findings from the install

**F1 — The action space is the keyboard, and that changes what an action is.**
`nethack.ACTIONS` is 121 members over **118 distinct bytes** (`+`, `"` and `$` each
appear twice, once as a command and once as a text character). Every letter is a
command *and* an answer: `d` is "drop" in the move loop and the inventory letter `d`
at `What do you want to eat? [d or ?*]`. `y` is both "move northwest" and "yes". So
naming actions by their command meaning — which is what every LLM-on-NetHack wrapper
does — is a lie at every prompt, and this harness names them by the key. Case is
significant: `h` steps west and `H` runs west. `?` is **not** in the action space, so
the in-game help is out of reach; `M-?` (the extended-command list) is in it.

**F2 — Menus and `--More--` are not skipped, and a key the game is not waiting for is
swallowed silently.** Measured: 3,000 consecutive compass keys from a fresh
`NetHackChallenge-v0` advanced the game **nine turns**, because at turn 9 a `--More--`
appeared and every subsequent key bounced off it. Nothing in the return value says so —
no error, no flag, the same shape of observation, and the clock simply stops. This is
the single most expensive fact about the environment and it dominates every floor
below.

**F3 — Reward is the in-game score delta, and the score is already on the screen.**
`NetHackScore._reward_fn` is `score(t) − score(t−1)` plus a time penalty; the Challenge
sets the penalty to zero and so do we (the default −0.01 per frozen step would make the
reward a comment on the interface — reading your inventory would cost points). NetHack
draws `S:` on the status line, so showing the reward gives away nothing the screen does
not. Craftax had to argue for showing reward; here it is free.

**F4 — Termination is death, quit, escape, ascension, the step cap, or the no-progress
abort.** `info["end_status"]` says only that something ended; `nethack.how_done()` is
NetHack's own end-of-game type and has **sixteen** values, eleven of which are ways to
die (`DIED`, `STARVING`, `CHOKING`, `DROWNING`, `TURNED_SLIME`, …). The harness keeps
the name and groups the deaths, because "starved on turn 2,391" is a different
diagnosis from "killed by a jackal".

**F5 — `NetHackChallenge-v0` refuses to be seeded, and that is the design's hinge.**
Its `__init__` overwrites `set_initial_seeds`, `set_current_seeds` and
`get_current_seeds` on the underlying game object with a function that raises, and its
`seed()` raises too — an anti-TAS measure for a competition where submissions could
otherwise memorise seeds. Consequence: **a run of it as registered can never be
replayed, and a run that cannot be replayed cannot be resumed.** Its parent
`NetHackScore` takes the same kwargs and accepts a seed, so the harness builds the
Challenge's configuration there (full `ACTIONS`, `character="@"`,
`allow_all_yn_questions`, `allow_all_modes`, zeroed penalties, the 10⁶ cap) and re-adds
the no-progress abort itself. A seed fixes the dungeon **and the character**: seed 0 is
a chaotic elven Priest, seed 1 a lawful female human Samurai, seed 2 a neutral female
human Healer, every time.

**F6 — Seeding is per-reset, not sticky.** `seed()` sets the seeds the *next* game will
be dealt from; the episode after that is dealt from wherever the RNG then is. Measured:
one seeded env reset three times gave a Samurai, a Monk and a Valkyrie. So every life
has to be seeded, which is exactly what makes the "same game on restart" default
possible — and is also the standing answer to F14.

**F7 — With reseeding off, `(seeds, keys)` is the whole record of a game.** `reseed=False`
disables NetHack 3.6's periodic reseeding with true randomness. Measured: 800 random
keys replayed from the same seeds gave bit-identical `tty_chars` at every step and an
identical reward sequence; a different seed diverged immediately. So the key history
`state.json` already keeps is a complete record, frames on disk are a cache rather than
an archive, and `--stint` / `--resume` / `--continue` all work exactly as they do in
`cc_craftax`.

**F8 — It is fast, and there is no compile to pay for.** 0.032 ms/step raw; 0.11 ms/key
through the wrapper including the screen decode and the life bookkeeping; `reset` 2 ms.
**Replaying a whole 3,000-key run with its deaths takes 0.33 s**, and a full
`act init --resume` round trip on a 40-key run — process start, game build, replay,
socket up — is 0.53 s. Craftax paid 25 s of JAX compile before its first action; this
pays nothing. A 30,000-key run resumes in about three seconds.

**F9 — Three observation channels, all native, and the pixel one is incomplete.**

* **tty.** `tty_chars` is the 24×80 terminal exactly as NetHack drew it: the message
  line, the map, and the two status lines. ~817 bytes a screen after trailing-space
  stripping, so a 3,000-key run is ~2.5 MB. This is the human channel and it is
  complete.
* **ansi.** `tty_colors` alongside it, re-emitted as SGR codes (~1.1 KB a screen).
  NetHack's colours carry information a player uses.
* **pixels.** `render_mode="pixel"` draws the map with NetHack's own tileset:
  **336×1264×3**, 0.02 ms a frame, ~4.3 KB as a PNG. It draws the **map only** — no
  message line, no status line, no menus — so a run given pixels alone could not read
  `You die...`. `NleGame` refuses that combination at construction.
* **symbolic.** `glyphs`, `chars`, `colors`, `specials`, `blstats`, `message` and the
  four `inv_*` arrays, as an `.npz` (~2.6 KB compressed).

**F10 — Human calibration, from the NetHack Learning Dataset** (Hambro et al., NeurIPS
2022 D&B). NLD-NAO is **1,511,228 games from ~48,454 people** on nethack.alt.org,
2009–2020 — incomparably better calibration than the six trajectories Craftax had:

| | NLD-NAO (human) | NLD-AA (AutoAscend) |
|---|---|---|
| games | 1,511,228 | 109,545 |
| median score | **836** | 5,422 |
| mean score | 127,218 (Zipfian) | 10,105 |
| **median keypresses** | **1,724** | 28,181 |
| median game turns | 3,766 | 20,414 |
| ascension rate | 1.46% | — |

Two readings. **The median human game is 1,724 keystrokes long**, so a 3,000-key budget
is above it — a run at this budget is not being cut off before a typical game would
have ended, it is being given about 1.7 median games' worth of keyboard. And **the
median human scores 836**, which is the number to beat before anything else; AutoAscend
needs 28,181 keys for its 5,422, so bot-level scores are not on the table at this
budget and should not be the target.

**F11 — Turns are bought with keystrokes at a variable rate.** Humans average 2.2 game
turns per keypress (F10: 3,766 turns over 1,724 keys). NetHack's own count prefixes and
multi-turn commands are why: `2` `0` `s` is **three keys and sixteen turns** of
searching (measured), `_` travels across a level for four, and a shift-direction runs
until something interesting happens. So the budget is not a turn budget, and an agent
that learns this gets several times more game out of it. This is the native analogue of
Craftax's `left*12`, except that it is the game's and not the harness's.

**F12 — The tombstone frame has no status line.** When a life ends NLE zeroes `blstats`
for the final observation, so a dead life's last self-report is 0 points on turn 0 at
depth 0. Every per-life number the harness keeps is a running maximum for this reason.
(This is the kind of thing that silently reports every run as scoring nothing.)

**F13 — Floors, measured** (`tools/baselines.py`, 3,000 keys, 5 seeds, restart on
death). All four play by exactly the rules a session plays by, and none reads anything
a session could not:

| policy | game turns | best life | mean/completed | lives | deaths | max dlvl |
|---|---|---|---|---|---|---|
| **noop** (`.` only) | 55–956 | **0** | 0 / 0 | 1 | 0 | 1 |
| **random** (all 118 keys) | 7–37 | **0** | 0 / 0–1 | 1–2 | 0 | 1 |
| **blind** (8 compass keys) | 32–172 | **0–4** | 0 / 0 | 1 | 0 | 1 |
| **wander** (compass + `cr`/`esc`/`y` + travel to `>`) | 195–5,732 | **0–158** | 0–52 | 1–6 | 0–5 | 1–4 |

The same four at **400 keys**, which is the pilot's budget and so the row a short run
is read against:

| policy | game turns | turns/key | best life | lives |
|---|---|---|---|---|
| noop | 55–304 | 0.14–0.76 | 0 | 1 |
| random | **2–9** | 0.01–0.02 | 0 | 1 |
| blind | 32–172 | 0.08–0.43 | 0–4 | 1 |
| wander | 194–530 | 0.49–1.33 | 0–52 | 1–2 |

The shape of this table is the finding. **Uniform-random over the real action space is
a *worse* floor than random movement**, because most of the keyboard opens something
modal and a random policy almost never closes it: 3,000 keys buy 7–37 game turns, and
400 keys buy *two*. Even `noop` outlives it. And the whole gap between `blind` and
`wander` — 0 points to 158, dlvl 1 to dlvl 4 — is three keys of interface handling.
Contrast Craftax, where random play collected 4–6 achievements: there the floor
measured the game, here it measures the terminal. A session's number has to be read
against `wander`, not against random — and the sharpest single comparison is
**turns per key**, where `wander` manages 1.1–1.3 against a person's 2.2 (F11).

**F14 — Quitting costs two keystrokes, and a re-rolled character would make that a
lottery.** `#quit` then `y` ends the game (`how_done() == QUIT`). It is a real key of
real NetHack, so unlike Craftax's `reset` — which this harness family removed because
it was an *addition* to a 43-action interface — it stays. What makes it harmless is
F6: every life of a run is dealt the same seed, so quitting to re-roll returns the same
character in the same dungeon. Had lives been dealt fresh characters, a run judged on
its best life would have rewarded quitting until a Valkyrie turned up — the exact shape
of the `reset left do` failure that cost Craftax's first pilot 2,200 actions, seen in
advance and designed out rather than discovered.

**F15 — NLE writes NetHack's own recordings.** With `savedir` set it writes a
`.ttyrec3.bz2` per episode (~3 KB per 300 keys) plus NetHack's `xlogfile`, which
records how each game ended and what it scored. `nle.scripts.ttyplay` replays them.
These go beside the environment, never into the workspace — the xlogfile is a spoiler
for the run's own outcome — and they are a ready-made archive for the replay page.

**F16 — The model has read about this game.** NetHack is thirty-eight years old, has a
wiki that answers everything, and its status line (`Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:9
Xp:1/0 T:1`) is recognisable to anything that has read about it. Every baseline we are
quoting was *written by* people who knew the game — AutoAscend is nothing but NetHack
knowledge. So recognition cannot be prevented and is not misconduct: the harness
withholds the *name* (F18, Decision F) and the audit measures when the session works it
out anyway.

**F17 — NLE hands out views into buffers it reuses.** `copy=False` is the default, so
an observation kept past the next step silently becomes the next observation. Anything
that stores one must copy it. This is pinned by a test, because the way it fails is
that two states compare equal and a bug looks like a finding.

**F21 — A chained run's bill is the sum of its sessions, and the report took the
last one.** One agent session emits exactly one `result` event; a stinted run is
several sessions appending to one stream. `absorb` assigned rather than summed, so a
three-session run reported its final session's cost as the run's — the void second
pilot printed **$15.57** for three sessions whose first alone cost **$31.77**.
Inherited from `cc_craftax`, where it was harmless because a run without `--stint` is
one session and the last result is the only result; it became wrong the moment a run
could outlive the session playing it, and it would have under-reported the whole
matrix. Fixed in `rig/agents.py` and in `tools/readout.py`, which now also says how
many sessions it summed. Codex keeps the assignment, with a comment: it reports usage
cumulatively *within* a session and fires every turn, so summing there would
over-count — a chained Codex run under-reports, which is now written down rather than
silent.

**F20 — NetHack asks the operating system what day it is.** It plays differently on
a full moon, on Friday the 13th, at night and at midnight, and with NLE's default
(`fix_moon_phase=False`) all four come from the real system clock. So `(seeds, keys)`
is not the whole record: **the same run replayed on another day is a different game**,
and so is a resume after midnight — which is precisely what a long `--stint` run does
to itself. `fix_moon_phase=True` derives the four from the seed instead, and with
seeds always set here there was never a reason for it to be off.

This is what voided the second pilot. The first divergence was `Be careful!  New moon
tonight.` appearing in a replay where the recorded run had no such line: the run began
at 22:55 on one date and was replayed on the next.

Pinned by the seed-to-phase mapping NLE documents — with the clock in charge seeds 0,
3, 5 and 49 all show whatever tonight happens to be; with it out of the way they show
new moon, full moon, nothing, and Friday the 13th.

**One cost, paid.** On a seed with a moon message NetHack says the moon *after* the
welcome, so the welcome is gone before the first observation and the character cannot
be read off the screen. It now comes from `^x` played in a **throwaway game dealt from
the same seeds** — twenty milliseconds, and the run's own budget and history are
untouched, because an action spent there would be an action the agent did not choose
and the record could not explain.

**F19 — NLE tells NetHack to write two files it never creates, and it broke the
first full run.** With `savedir` set, `nle/env/base.py` points NetHack's `record` and
`logfile` at `<savedir>/nle.<pid>.record` and `.logfile`, while its own *"Touch files,
so lock_file() in files.c passes"* loop creates `perm`, `record` and `logfile` in the
**temporary** var directory and only `xlogfile` at the prefix. So the two files
NetHack writes **when a game ends** are missing, and the end of every life printed

    Cannot open file <savedir>/nle.<pid>.logfile.  Is NetHack installed correctly?

into the message area. Two consequences, and the second is the serious one:

1. The agent is handed a harness error to interpret at the worst moment it could get
   one — the screen where a life ended, which is the only view of that state.
2. **The message carries the process id**, so the same seed and the same keys stopped
   producing the same screens. F7 held only until the first death. A resume is a new
   process, so `--stint` was handing the next session a game that was not the one the
   record described.

It cost the first 3,000-key pilot, which is void: 2,012 keys over four sessions and
six lives, of which only the first life is a measurement of anything. The record says
the lives scored `[191, 134, 159, 602, 286, 647]`; a clean replay of the same history
scores `[191, 134, 159, 314, 24]` and reaches five lives, because sessions 2–4 each
resumed into a divergent game and carried on from there.

**How it was caught.** `tools/readout.py` refuses a replay whose per-life scores
disagree with the record. That check was written as a guard against a class of bug
nobody expected to see, and it fired on its first real outing — which is the argument
for writing it. Nothing else would have noticed: the run looked healthy, the launcher
reported handovers, the scores went *up*, and a page built from the record would have
shown a perfectly plausible game.

Fixed by creating the three files at the prefix NLE computes, and pinned by two tests
— a life ending while recording says nothing about the harness, and a run with a death
in it replays bit-exactly while recording, and identically to the same run with
recording off.

**F18 — The game names itself, and so did four other things.** A brief that withholds
what is being played is worth nothing if the workspace announces it, and this one did,
in four places — all now closed, all tested by grepping a played workspace for the
word:

1. **The opening screen.** `Hello Agent, welcome to NetHack!  You are a chaotic elven
   Priest.` — the first observation of every run. `v` and `#version` say it again
   (`Unix NetHack Version 3.6.7`).
2. **`state.json`**, which carried `"variant": "nethack"` in a file sitting beside the
   log the session is told to read.
3. **The workspace shims.** `cat act` printed
   `exec "<...>/cc_nle/nle-code/.env-venv/bin/python" "<...>/cc_nle/nle-code/act.py"`.
4. **The launch root**, `~/nle-runs/...`, which is the session's own working directory
   and therefore its `pwd`.

The fixes: (1) a **redaction** — the seven characters of the name are replaced by
`*******` in the tty, ansi and symbolic-message channels, same length so nothing on an
80-column screen moves. This is the one place the harness alters what the package
produces; it changes no dynamics (pinned: identical score, turns, depth and cells with
it on and off) and it withholds rather than invents, so a session that finds it can see
that something was withheld rather than being told something false. (2) `variant` and
`blind` move to the record beside the environment, and `pick_up` reads what the run
*is* from there. (3) The shims are written in terms of `<launch>/.bin/`, which holds
neutral wrappers — wrappers and not symlinks, because a venv's `python` reached through
a symlink lands outside its own venv and imports nothing (measured). (4) The launch
root defaults to `~/agent-runs`.

**None of that makes NetHack unrecognisable**, and the plan should not pretend
otherwise: a status line reading `Dlvl:1 HP:12(12) AC:9` next to a `--More--` is
unmistakable. What it buys is that recognition has to come from *the screen* rather
than from the plumbing — which is exactly what `named_the_game` is measuring — and that
a session cannot coast on a label it was handed before it looked at anything.

---

## 1. The decisions

Six. The first two are genuinely open; the rest follow from the findings.

### Decision A — the Challenge's configuration, seeded

The registered `NetHackChallenge-v0` is what the published leaderboard is on, and it is
what we want: the full keyboard, menus left alone, a rolled character. What we cannot
take is its refusal to be seeded (F5), because that costs the run's replayability, its
resumability, and any hope of a matrix in which two conditions face the same game.

So: **the Challenge's configuration, built on the parent class that accepts a seed.**
Every kwarg is copied from `tasks.NetHackChallenge`; the no-progress abort is re-added
by hand. The two deviations from the registered environment, stated plainly:

1. **The seeds are set**, from the harness's `--seed`. Every published NLE result that
   wanted reproducibility does the same; the competition's ban exists to stop
   submissions memorising specific seeds, which does not apply to a session that has
   never seen one before and keeps no weights.
2. **`reseed=False`**, which turns off NetHack's periodic reseeding with true
   randomness. Without it F7 does not hold.

`--variant score` gives the registered `NetHackScore-v0` task instead — 23 actions,
menus auto-skipped — which is the configuration the NLE paper's RL numbers are on and
a genuinely different game to play. Wired, and off.

### Decision B — an action is a keystroke, named by the key

This is the decision the whole interface rests on, and F1 forces it. The alternative —
naming actions by their command meaning, as BALROG's wrapper and every other
LLM-on-NetHack setup do — is not a translation but a change of environment: it hides
that `y` is both "northwest" and "yes", it cannot express an inventory letter, and it
turns the modal interface into a flat action space that NetHack does not have.

So the token vocabulary is the key table: 118 tokens, printable keys as themselves
(`k`, `>`, `*`, `$`), the seven control keys as `esc`, `cr`, `^d`, `^o`, `^r`, `^t`,
`^x`, and the meta keys under NetHack's own extended-command names (`#pray`, `#quit`,
`#chat`). Case is not folded. `h*12` repeats, as in the siblings; the game's own count
prefix (`2` `0` `s`) is available too and is the cheaper of the two (F11).

The cost of this decision is that the agent must handle `--More--`, menus, `[yn]`
prompts and the `getlin` line itself. That is not an accident of the port — it is 90%
of the gap between the `blind` and `wander` floors (F13), and it is the part of playing
NetHack through a terminal that a language wrapper deletes.

### Decision C — the observation channels are a switch, and it starts on `tty`

Four channels (F9), any combination, written per action, with the log naming what it
carries so a run is self-describing. **Now: `--obs tty`** — the 24×80 screen, which is
exactly what a person playing NetHack sees, complete, and 817 bytes.

`pixels` is available and interesting (it is the tile view a graphical NetHack gives),
but it draws the map only, so it is refused as the sole channel and belongs beside
`tty` in a VLM condition. `symbolic` is the network's view. `ansi` is `tty` plus the
colours, and is the one to add first if a session turns out to be confusing monsters
that differ only by colour.

### Decision D — episode structure, and the budget

The environment's own: a life runs until NetHack ends it. On an ending the harness
deals **the same seed again** (Decision A + F6) and keeps spending the same budget;
`--fresh-world` rolls a new character per life instead, which is the Challenge's own
behaviour and the generalisation setting.

Dealing the same game again is Craftax's default and it earns its keep twice here:
what the session learned about this dungeon and this character still holds, and the
two-key `#quit` stops being a re-roll lottery (F14).

**Budget: 3,000 keystrokes.** The justification is F10 and it is unusually solid: the
median human game on nethack.alt.org is **1,724 keypresses** and scores **836**. A
3,000-key budget is 1.7 median games, spent as one life or several, and it is a tenth
of what AutoAscend's median game takes. At the human turn rate (F11) it is ~6,600 game
turns; at the `wander` floor's rate it is fewer. What it buys is the early dungeon —
the first handful of levels, the first real fights, the first equipment — and not the
mid-game.

The actuator records the score after every action, so a 3,000-key run *contains* the
1,000-key result and the score-against-keys curve comes out of any run for free.

### Decision E — what the log carries

`cc_humanrl`'s rule holds unchanged: the log tells the agent about the protocol, never
about the world. Per action: the index, the budget, the key played, the reward, and
the path of each enabled channel. Nothing computed from a screen; nothing named off
one.

What is kept out of the workspace is narrower than in the siblings, and deliberately
so. The score, the depth, the experience level and the hunger state are all on
NetHack's own status line — withholding them would be withholding the game. What stays
in `result.json` beside the environment is the **run's view of itself**: the table of
lives, which was best, the mean over completed ones, the ttyrecs and the xlogfile. A
session that could read its own league table would play it.

### Decision F — the brief withholds the game, and so does everything else

The sibling default, kept: *"You are playing a game you have never seen before.
Something in it is you."* What is different here is that keeping quiet is not enough —
the game introduces itself on the first screen and the plumbing named it in three more
places (F18). All four are closed, and a test greps a played workspace for the word.

The honest limit is stated in F18 and belongs in any write-up: this is a fence against
*accidents*, not against recognition. A model that reads one status line knows what it
is playing. What the blind condition buys is that the knowledge has to arrive through
the observation channel, where the audit can see it arrive (`named_the_game`), rather
than being handed over in the brief before the session has looked at anything.

`--opening named` is the ablation: the same run, one paragraph different, with the
redaction off so the brief and the screen agree. The pair is a real experiment — how
much of an agent's score on a game this well documented is recall and how much is
reading — and it is one flag rather than a second harness.

### Metrics, harness-side

| metric | how |
|---|---|
| `best_episode` | the best single life's score — **the headline**, against 836 (median human) and 5,336 (AutoAscend) |
| `mean_episode` | mean over the lives the game ended, with `episodes_completed` as its n — the Challenge's own unit |
| `episode_scores` | score per life in order: the within-run learning curve |
| `max_depth`, `max_xplevel` | how deep and how strong — the axis BALROG's progression metric is built on, and where LLM agents stop (dlvl 3, xp 4) |
| `turns` | game turns bought with the keystroke budget (F11) — an efficiency measure with no analogue in the siblings |
| `endings`, `roles` | how each life ended in NetHack's vocabulary, and what character it was dealt |
| `unique_cells` | distinct (branch, level, position) visited — exploration |
| `obs`, `opening`, `variant`, `seed` | the conditions, on every report |
| cost, tokens, turns, compactions | from the stream, as in the other three harnesses |

---

## 2. What to build

Sibling of `cc_craftax/`, same shape, because the shape is the part that already works.

```
cc_nle/
  nle-code/                   # its own git checkout, gitignored from bai
    act.py                    # actuator + daemon   <- port, contract unchanged
    run.py                    # session launcher    <- port + the opening switch
    nle_game.py               # the craftax_game.py analogue
    GAME.md / PROMPT.md       # brief + doctrine
    rig/{agents,audit}.py     # agents unchanged; new audit table
    tools/{baselines,make_*_venv}.sh|.py
    tests/                    # 109
    .agent-venv/              # numpy + Pillow, no nle
    .env-venv/                # nle 1.3.0 — the daemon's interpreter, never the agent's
```

### Reused unchanged
`run.py`'s workspace creation, isolated `CLAUDE_CONFIG_DIR`, credential-lifetime rule,
`claude -p` with the allow/deny lists, stream capture, `Report`, replay/rotation,
label assignment; `rig/agents.py`; the daemon-over-unix-socket design; the `logs.txt`
block format; `--plan` recording; batch-stops-on-event; log rotation; the stint/resume
machinery.

### New

**`nle_game.py`** — owns one game, its seeds, its channels and its lives. `play(key) →
(reward, alive)`, plus `restart()`, `screen()`, `ansi()`, `frame()`, `symbolic()`,
`observe()` and the score readers. The responsibilities are the findings: the
seedable Challenge reconstruction (F5), per-life seeding (F6), the key table (F1), the
no-progress abort (F4), NetHack's own ending names (F4), the maxima that survive the
tombstone (F12), the pixels-alone refusal (F9).

**`act.py`** — same contract. Three differences from the Craftax port, all in the
module docstring: an action is a keystroke, a life can be ended by the player and that
is the game, and there is no denominator.

**`rig/audit.py`** — a new table. The line stays where `cc_craftax` put it — reaching
outside is a finding, knowledge is not — and two patterns are NetHack's own: **a second
game** (`nle.scripts.play`, `gym.make`, a system `nethack` binary — an oracle the agent
could consult offline for as many keystrokes as it likes) and **the game's data files**
(`nethackdir`, `nhdat`, `monst.c` — the wiki without the network). The word "nethack"
is deliberately *not* a pattern: the harness's own path carries it and so does the
brief.

**`tools/baselines.py`** — the four floors of F13.

### Build steps

* **M0 — pin and scaffold.** *Done 2026-09-09.* `nle==1.3.0` into `.env-venv` (a clean
  wheel install, no compilation, unlike Craftax's 776 MB of JAX); `.agent-venv` with
  numpy and Pillow and a check that it cannot `import nle` or `import gymnasium`.
  `tests/test_env.py` (12 tests) stands in for a vendored commit: it pins the version,
  the 121/118 action counts, the absence of `?`, the 24×80 screen, the tile shape, the
  Challenge's seed refusal, the parent class's acceptance, bit-exact replay, that a
  seed rolls the character, that seeding is not sticky, that `--More--` swallows
  everything, and that the agent's interpreter is fenced.
* **M1 — `nle_game.py` and its tests.** *Done 2026-09-09, 20 tests.* Settled while
  building it: the vocabulary is the 118 distinct keys and not the 121 actions; case is
  not folded; per-life seeds are `(seed, seed+7919)` offset by the life index only when
  `fresh_world`; every per-life stat is a maximum (F12); a channel combination cannot
  change how the game steps (pinned); `pixels` alone is refused.
* **M2 — `act.py` and its tests.** *Done 2026-09-09, 30 tests.* A life ending is not
  the run ending; the block holds the tombstone and then what the next life began in;
  `#quit`/`y` is the ending a test can reach in constant time; `state.json` carries no
  league table; the preamble names keys and explains nothing; `--stint`, `--resume`
  and a rebuild that replays across a death boundary all pinned.
* **M3 — brief, doctrine, launcher wiring, audit.** *Done 2026-09-09, 47 tests.* The
  `blind`/`named` switch is one paragraph of one file and a test pins that the rest of
  the two briefs is byte-identical. `PROMPT.md` names no environment, as in the
  siblings — its one NetHack-shaped addition is the modal-interface paragraph, written
  without naming a mechanic. The audit's false positives (a session printing its own
  `act` shim, its own launch paths, its own notes) are pinned clean. Dry run builds
  workspaces, starts games and records the label mapping under `.rig/`.
* **M3b — the blind condition made real.** *Done 2026-09-10, 119 tests.* The four
  leaks of F18 found and closed: the game's own name redacted from the text channels,
  `variant` and `blind` moved out of the workspace and into the record (so `--resume`
  now reads what the run *is* from the env-dir), the shims routed through
  `<launch>/.bin/`, and the launch root renamed. The check that matters is
  end-to-end: play a workspace, `grep -ri nethack` it, get nothing.
* **M4 — floors.** *Done 2026-09-09, F13.* The **ceiling is deliberately not built**,
  and this is a departure from `cc_craftax`. There, a 115-action scripted route to iron
  was writable in an afternoon and became the environment's regression test. NetHack
  has no such route: a probe that travels to `>` whenever one is visible reaches dlvl 1
  and nothing else, because on turn 1 the down staircase has not been discovered and
  finding it *is* the game. The nearest honest ceilings are external — AutoAscend's
  5,422 and the human median's 836 — and the regression test's job is done by the
  bit-exact replay tests instead.
* **M5 — pilot.** *Done 2026-09-10.* One session, `nethack:0`, blind, `--obs tty`,
  budget **400** rather than 3,000 — a short run first, because the failure worth
  finding cheaply was F2, and a session that never works out `--More--` would have
  burned $90 to say so.

  **Result: 402 points in 400 keystrokes, one life, never died, Dlvl 6, Xp 2, 757
  game turns, $12.81, 31 minutes.** Audit clean.

  | | |
  |---|---|
  | this run | **402 points / 400 keys** = 1.01 points a key |
  | median human game (F10) | 836 points / 1,724 keys = 0.49 a key |
  | `wander`, same budget (F13) | 0–52 points, Dlvl 1–2 |
  | best LLM run ever recorded (BALROG) | Dlvl 3, Xp 4 |

  The curve: 7 points at 100 keys, 211 at 250, 402 at 400 — accelerating, because
  the session worked out that NetHack's score is roughly `gold + experience + 50 ×
  (deepest level − 1)` and that **descending is worth 50 points a level**, then
  spent the rest of the budget diving.

  **The four questions M5 was for, answered.**
  1. *Did it handle the interface?* Yes. Four keystrokes out of 400 (1.0%) were
     thrown into a `--More--` that could not take them, the longest such run was
     three — and its `notes.md` says *"Four keystrokes swallowed whole by an
     unnoticed --More-- while attacking. Always check the message line before
     batching attacks."* It noticed, which is the thing that could not be assumed.
  2. *What did it buy with the budget?* **1.89 game turns per keystroke**, against
     the human 2.2 (F11) and `wander`'s 1.1–1.3. It found the mechanisms itself and
     wrote them down: count prefixes (`8` `h` is two keys and eight squares), `G`+dir
     runs, and travel — `_` `>` `.` crosses a level for three keys, `_` `.` re-uses
     the cached destination for two, and `_` `x` `.` travels to the nearest
     unexplored square.
  3. *How far?* Dlvl 6 — twice the deepest level any LLM agent has been reported to
     reach, in 400 keystrokes.
  4. *What does it cost?* **$0.032 a key** (145 tool calls, 2.8 keys each, 286 agent
     turns, no compaction), so 3,000 keys is ≈ **$96 and ≈ 4 hours**. The rate
     improved sharply after the opening: the first 90 keys took 25 minutes and the
     remaining 310 took six.

  **What the notes show.** It recognised the game almost immediately — the audit
  caught `"…establishing that hjklyubn move as in nethack"` in a `--plan` at agent
  turn 10, from the screen and not from the plumbing (F18), which is exactly the
  measurement the blind condition exists to take. What followed was not recall
  though: it *derived* the score formula from two observations (+3 for three gold,
  +4 for a jackal), measured that `autoopen` was on, established that diagonal moves
  through real doorways are forbidden but doorless gaps are fine, found that the tty
  dump does not show the travel cursor so cursor nudges are blind, worked out that
  prayer only heals after roughly T:300, and priced Elbereth at eleven keystrokes and
  declined to use it. It also kept a map of Dlvl 1 with a ten-key route from the
  upstairs to the downstairs, explicitly *"for reuse after a death, the level is the
  SAME each life"* — the harness's own restart rule, read off the brief and turned
  into an asset.

  **What this changes about the budget.** The plan sized 3,000 keys to buy "the early
  and middle dungeon" from a standing start. At the rate this run set it would buy
  far more, and the binding constraint stops being budget and becomes **survival**:
  the run never died in 400 keys on Dlvl 1–6, but the levels that pay 50 points each
  are also the ones that kill. So 3,000 stands, and the number to read off the full
  pilot is not the score at 3,000 but *when the first life ends*.

* **M5b — the full-budget pilot.** *Done 2026-09-11, on the third attempt. The
  first two were void (F19, then F20) and the fixes are what made this one a
  measurement.* `nethack:0`, blind, `--obs tty`, budget 3,000, `--stint 1000`.
  **Verified: the history replays exactly**, across two handovers and four deaths.

  **Result: best life 795 points, Dlvl 10, in 480 keystrokes — and still alive when
  the budget ran out.** Mean 350.25 over the four lives the game ended. 4,549 game
  turns, three sessions, 136 minutes, **$71.93**, audit clean.

  | life | keys | score | dlvl | ended |
  |---|---|---|---|---|
  | 1 | 841 | 378 | 6 | died |
  | 2 | 378 | 243 | 4 | died |
  | 3 | 850 | 438 | 4 | died |
  | 4 | 451 | 342 | 4 | died |
  | **5** | **480** | **795** | **10** | budget |

  | | points | keys | points/key |
  |---|---|---|---|
  | **this run's best life** | **795** | **480** | **1.66** |
  | median human game (F10) | 836 | 1,724 | 0.49 |
  | AutoAscend median (F10) | 5,422 | 28,181 | 0.19 |
  | `wander`, 3,000 keys (F13) | 0–158 | 3,000 | ≤0.05 |

  So the best life reached **95% of the median human game's score at 3.4× the
  median human's rate per keystroke**, and was cut off by the budget rather than by
  death. Dlvl 10 is the depth *one in twenty* of AutoAscend's Valkyrie games
  reached, and more than three times the deepest level any LLM agent has been
  reported to reach (BALROG's best single run: Dlvl 3).

  **The curve across lives is the finding.** 378 → 243 → 438 → 342 → **795**, and
  the last life reached deeper than any before it in *half the keystrokes* of the
  first. This is the thing no RL baseline has an analogue for, and the session's own
  notes say exactly what carried across: *"What made the difference was, in order:
  killing things instead of only descending … taking a **pick-axe** off a dwarf and
  digging straight down: 4-7 keystrokes per dungeon level instead of 30-80 … using
  prayer as a full heal."*

  **What it worked out, measured rather than recalled.** It derived the score
  formula — `4 × experience + gold + 50 × (deepest dlvl − 1)`, and 1000 a level past
  Dlvl 20 — and checked it: *"every kill moved S by exactly 4x the change in the
  Xp:n/NN number; each new dlvl added exactly 50."* From that it drew the conclusion
  that decides the run: *"killing things you can safely kill is ~10x more
  score-efficient than descending"* — a gnome is +28 for three keystrokes, a whole
  dungeon level +50 for fifty to eighty. It found that **digging erases a dust
  Elbereth**, which is a subtle interaction it paid ~40 keys to learn; that an
  adjacent monster interrupts a dig every turn; that three of its four deaths were
  the same monster (*"an `h` in the Mines carrying a broad pick"* — a dwarvish
  mattock, 3d6); and it **climbed back up out of the Gnomish Mines** on purpose,
  *"because the Mines dead-end at ~Dlvl 12 while the main dungeon runs past 25 and
  only the main dungeon can pay the 1000-per-level rate."* It wrote five tools
  (`play.py`, `dive.py`, `digdown.py`, `explore.py`, `show.py`) and kept a list of
  bugs it had already fixed in them so a successor would not reintroduce them.

  **Where the budget went.** 1.52 game turns a key, below the human 2.2 and below
  its own 400-key pilot's 1.89 — because this run fought, and fighting is keystrokes
  at a prompt rather than movement. 2.4% of keys (73) were thrown into a `--More--`
  that could not take them, longest run 8, up from 1.0% in the short pilot; its
  notes name the cause (*"an adjacent monster interrupts the dig every single
  turn"*). 54 distinct keys, 4.0 keys a tool call, no compaction.

* **M6 — the matrix.** *Next, and now unblocked.* N seeds at 3,000 keys each. A seed is a character (F5) and the roles differ enormously in
  survivability, so the matrix has to be large enough to say something about the
  distribution rather than about one Priest: 8–12 seeds at ≈$96 each, with the roles
  reported beside the scores.
* **M7 — the replay page and the readout.** *Done 2026-09-10, 131 tests.*
  `tools/readout.py` replays a run and reports what no file kept: the curve at every
  milestone, and the three things the pilot asks — keys thrown into a prompt that
  could not take them (with the longest such run, which is what says whether the
  session *noticed*), game turns per keystroke against the human 2.2, and best life
  against keys spent. `tools/replay.py` builds the scrubber, and this is the port
  where the page gets *smaller*: a step changes about two rows of eighty characters,
  so a run inlines whole with its colours, and the test folds the deltas in Python
  the way the browser folds them in JS and compares against the screens the actuator
  wrote at the time. Both are built from the keystroke history rather than from the
  files a run happened to write, so they work on any `--obs` and on a run whose
  screens `--again` rotated away.

---

## Risks

* **The interface eats the run.** F2 and F13 together say a policy that mishandles
  modal screens gets nowhere at all, and unlike a dropped action it is silent. The
  pilot's first job is to check that the session notices. If it does not, the fix is
  not to auto-skip `--More--` — that is a different environment — but to read the
  failure as a result about coding agents and terminals.
* **Recall without execution.** The opposite failure, and the more interesting one: a
  session that can recite the Elbereth trick, the early-game Valkyrie plan and the
  hunger clock, and still cannot get any of it through 3,000 keystrokes of a modal
  interface. That gap is the thing this harness measures and the write-up should be
  built around it.
* **Blindness is a delay, not a fence (F18).** The name is withheld and the plumbing
  no longer leaks it, but one status line identifies the game. Expect
  `named_the_game` to fire in most runs, and report *when* it fired and off what —
  a session that recognised NetHack at action 3 and a session that never mentioned it
  are different results, and neither is misconduct. If a run is wanted where
  recognition genuinely cannot happen, that is a different environment, not a flag.
* **Character variance.** Score variance across roles in NetHack is enormous — larger
  than across seeds within a role. Eight seeds is a small sample of thirteen roles, so
  no mean is worth quoting without the roles beside it, and a fixed-character condition
  (`--character val-hum-neu-fem`, the usual choice) may be needed to say anything
  crisp.
* **Comparability.** Our number and AutoAscend's are both "score on the Challenge's
  configuration", but ours is on ~3,000 keys and its median game is 28,181. Say so in
  the table. The human median (836 over 1,724 keys) is the fair comparison and is the
  one to lead with.
* **Token cost.** Screens are cheap (817 bytes) but there are 3,000 of them, and the
  PROMPT's "read them with code" is load-bearing. Watch the compaction counter, as in
  Craftax.

---

## 3. What this gets us

1. **An agentic number on NetHack in the benchmark's own units**, beside the human
   median (836) and the Challenge's leaderboard (5,336.5), and beside the only existing
   LLM number — BALROG's 1.57% progression, whose best run in the whole benchmark
   reached dungeon level 3.
2. **The first LLM-agent number on the *unwrapped* interface.** Every prior
   LLM-on-NetHack result renames the keys into English. This is the same game through
   the terminal, and the gap between the two is itself a result.
3. **A within-run learning curve** — score per life across the lives of one run, in a
   game where the character and dungeon are held fixed. The agent carries notes across
   deaths where a policy carries weights, and NetHack is a game where knowing the map
   is worth a great deal.
4. **The named/blind pair.** How much of an agent's NetHack score is recall and how
   much is reading the screen, on the same seeds, one paragraph apart.
5. **Continuity.** Fourth benchmark, same harness, same doctrine. The marginal cost of
   the fifth is smaller again — and this one was cheaper than Craftax by the cost of a
   JAX dependency.

---

## 4. Deferred, deliberately

* **MiniHack.** The same package's curriculum of small, goal-directed rooms, with its
  own published baselines. It is a different benchmark with a different unit and would
  be a fifth harness, not a flag on this one — but the actuator would need nothing new.
* **The other channels.** `ansi`, `pixels` and `symbolic` are wired and off (Decision
  C). The pixel condition is a genuine VLM experiment: the tile view is what a
  graphical NetHack shows a person.
* **The `score` variant.** Wired. It is the configuration the NLE paper's RL baselines
  are on, and running it would put us beside those numbers rather than the Challenge's
  — but it is a friendlier interface and the friendlier interface is not the point.
* **A fixed character.** `--character` is one kwarg away and would cut the variance in
  Risks. Held back because a rolled character is what the Challenge does.
* **Checkpoint/rewind as an agent-facing instrument.** F7 makes it free and it would be
  a genuinely new capability — but it changes the game rather than measuring the agent.
  (NetHack's own `S` save is in the action space and ends the game, as it should.)
* **NLD as a comparison set.** 1.5M human games are downloadable and would support a
  score-against-keystrokes curve rather than the two points in F10. Heavy, and the two
  points are enough to justify the budget.
