# 7xf97 — TEST50 held-out pool (51 scored transitions)

Config: whitelist=`left,right,up,down,noop,click`; `keep_action_params=FALSE` (click collapses
to the bare verb; the ID label is the verb, so click targets must move the Sun visibly).
Verified with `T.verify_pool('prototypes/perc_invdyn/clean_data3/7xf97/test50', 'left,right,up,down,noop,click', context_k=9)`
-> **51 scored target transitions** (within the 50±2 spec).

All slices are verbatim rows from four FRESHLY DRIVEN seed-0 trajectories
(`autumn_drive.py 7XF97 ...`), none reuse the train/train_regen action sequences or states:

| drive | theme | how it differs from train_regen |
|---|---|---|
| A | cloud 13→7, drop + green grow at col 7 | train watered only col 13 |
| B | sun full round-trip x0→13→0 with BOTH edge bounces in the OPEN grid; cloud transits 13→2→13 | train's right-edge bounce happened hidden under the cloud; here the cloud is parked at origin 2 during the right-trip and back home during the return, so both bounces and all reversal clicks are fully visible |
| C | tree grown at col 5 (green 14, green 13, purple 12), 4th drop despawns on the purple, 5th drop at col 6 exits off-grid; left/right/click injected mid-fall | train's tree was at col 13; off-grid despawn NEVER occurs in train; compound click-during-fall is new |
| D | blocked-vs-unblocked watering at col 9 (sun cols 6-8 overlaps cloud col 8 -> blocked ×2 driven, 1 scored; sun moved to x=12 -> same leaf grows) | train's blocked case was col 13 with sun origin 11; here same-column contrast is driven twice with the outcome flipping on sun position only |

## 1. Core dynamics (from dynamics.txt)

D1 left(cloud -1) · D2 right(cloud +1) · D3 down spawns Water below cloud origin (occluded
inside the 3-tall cloud for 2 steps) · D4 click-on-Sun moves Sun (left if movingLeft else
right) · D5 up = no-op · D6 noop = no action effect · D7 per-step Water gravity ·
D8 Sun edge-bounce flips movingLeft at x=0 / x=13 · D9 Water despawns on hitting a Leaf ·
D10 watering a GREEN leaf grows a new Leaf one above (mediumpurple at row 12) unless the Sun
overlaps the Cloud · D11 spawn/despawn bookkeeping (off-grid removal) · D12 none.

## 2. Slices (15 episodes, 66 rows, 51 pairs)

| ep | src steps | scored pairs (action -> change) | dynamics / role |
|---|---|---|---|
| 0 | A 7..11 | L gray-1 ; **down** NC ; noop NC ; noop **blue+1 spawn** | D1; D3 cause-in-window drop at col 7 |
| 1 | B 4..7 | L ; L ; **up** NC | D1 mid-grid; D5 (aliased, see §4) |
| 2 | B 30..34 | **clickR** gold+1 (arrive x=13) ; noop NC (**bounce**) ; **clickL** gold-1 ; clickL | D4+D8 right-edge: same click verb, direction reverses across the bounce inside ONE slice |
| 3 | B 41..45 | R gray+1 ; R gray+1 gold-3 ; **up** NC ; R gray+1 gold-3 | D2 (incl. cloud occluding the sun — displacement still legible); D5 |
| 4 | B 63..67 | **clickL** gold-1 (arrive x=0) ; noop NC (**bounce**) ; **clickR** gold+1 ; clickR | D4+D8 left-edge bounce, fully visible (cloud parked at 12-15) |
| 5 | C 16..19 | noop **blue falls** ; **left**+fall ; **right**+fall | D7 gravity is action-independent (fires on noop AND on moves) |
| 6 | C 27..32 | **down** NC ; noop NC ; noop **spawn** ; noop fall ; **click**+fall gold+1 | D3 col-5 drop; D4 compound click-during-fall |
| 7 | C 39..42 | noop fall ; noop **blue-1 green+1** ; noop NC | D10 grow on a REGROWN leaf -> (13,5), tree climbing |
| 8 | C 53..56 | noop fall ; noop **blue-1 mediumpurple+1** ; noop NC | D10 grows into row 12 = mediumpurple, (12,5) |
| 9 | C 66..69 | noop fall ; noop **black+1 blue-1** ; noop NC | D9 despawn on a non-green (purple) leaf, NO grow |
| 10 | C 84..87 | noop fall ; noop fall (into row 15, col 6) ; noop **black+1 blue-1** | D7/D11 despawn OFF-GRID (no leaf in an even column) — never occurs in train |
| 11 | D 15..18 | **down** NC ; noop NC ; noop **spawn** | D3 col-9 drop; window shows sun (cols 6-8) overlapping cloud (col 8) |
| 12 | D 28..31 | noop fall ; noop **black+1 blue-1, green stays 8** ; noop NC | **D10 near-miss (BLOCKED)**: water lands on the green leaf (15,9) but sun-over-cloud -> despawn only |
| 13 | D 58..61 | **down** NC ; noop NC ; noop **spawn** | D3; sun now at x=12, clear of the cloud |
| 14 | D 71..74 | noop fall ; noop **blue-1 green+1** ; noop NC | D10 positive on the SAME leaf column 9 that was blocked in ep12 — only the sun differs |

## 3. Coverage map (scored targets per dynamic) + action histogram

By verb (clicks collapse): **noop 30, click 7, left 4, right 4, down 4, up 2** (= 51).

| dynamic | positives | negatives / contrast |
|---|---|---|
| D1 left | 4 (ep0, ep1×2, ep5 compound) | NC noops (cloud never drifts on its own) |
| D2 right | 4 (ep3×3 incl. 2 occluding the sun, ep5 compound) | direction contrast vs the 4 lefts |
| D3 down-spawn | 4 downs (ep0,6,11,13) + 4 spawn-visible noops | 11 NC noops that do NOT spawn; irregular action gaps kill step-clocks |
| D4 click | 7 (4 gold+1, 3 gold-1; ep6 during a fall) | bounce NC noops + 11 NC noops: sun moves ONLY on click |
| D5 up | 2 (ep1, ep3) | inherently NO_CHANGE — see §4 |
| D6 noop | 11 NC + 19 passive-event noops | up/down NC pairs are the near-miss |
| D7 gravity | 9 pure falls + 3 compound (left/right/click during fall, ep5/ep6) | NC noops with no water in the air: "noop moves water" is false |
| D8 edge-bounce | both edges: ep2 (x=13) and ep4 (x=0), bounce noop + reversal clicks in the SAME slice | click direction is conditional on the hidden flag, not fixed |
| D9 leaf-despawn | ep9 (purple hit), ep12 (blocked hit); every D10 grow also removes the water | vs D10 grows: same blue-1 cue, different outcome |
| D10 watering-grow | 3 (ep7 green regrown, ep8 purple@row12, ep14 green post-blocked) | ep12 BLOCKED near-miss on the same col-9 leaf as ep14 |
| D11 spawn/despawn | spawn = D3; despawn off-grid ep10 (new vs train) | leaf-despawn vs bounds-despawn distinguishable by row |
| D12 termination | n/a (none defined) | — |

Contrastive negatives: 11 NC noops + 2 up NC + ep12 blocked + ep9 purple-hit ≈ 15/51 ≈ 29 %
of the pool (spec target 20-30 %).

## 4. Residual unidentifiabilities / uncoverable items (documented, not hidden)

- **`up` is ID-identical to `noop`** (no handler): both NO_CHANGE. 2 scored ups kept for verb
  coverage; an oracle can only answer "a no-op". Inherent to the game; same caveat as train.
- **`down` is NO_CHANGE at its own step** (spawn occluded inside the 3-tall cloud). All 4 downs
  use the cause-in-window construction: the visible spawn (blue+1 at row 3) is 2 steps later in
  the SAME slice, so the window disambiguates; at the pair level down aliases noop/up.
- **D8's flag is hidden state**: the bounce itself is a NC noop; it is scored via the
  click-direction reversal around it (ep2, ep4). Only 2 bounce events exist (one per edge, each
  needs a full cross-grid trip); requirement of >=4 is met by the 4 reversal-adjacent clicks,
  not 4 bounce events.
- **D10 has 3 grow positives** (not 4): the 4th D10-targeting pair is the ep12 blocked
  near-miss. The 3 positives are maximally varied (regrown leaf, purple-at-row-12, post-blocked
  original leaf, at 2 different columns); a 4th would have duplicated one of these kinds.
- **click-on-empty-cell (D4 no-op branch) is not scored**: under keep_action_params=FALSE it
  would be a NO_CHANGE pair labeled `click` — unanswerable even for an oracle. Excluded per the
  observability requirement.
- **noop share is 59 %** (30/51): all passive dynamics (gravity, spawn-visible, grow, despawn)
  are noop-labeled by construction, matching the train distribution (58 %). The movement/click
  verbs are internally balanced (4/4/4/7, up 2).
