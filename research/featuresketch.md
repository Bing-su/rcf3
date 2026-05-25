# FeatureSketch

## Goal

FeatureSketch is an online anomaly detector for streams whose schema is not
fixed. Each event is represented only by its currently observed features. New
feature names may appear at any time, and previously common feature names may
stop appearing. The public API keeps algorithm parameters internal.

The public shape is deliberately small:

```text
detector = Detector()
score = detector.score(features)
detector.update(features)
```

`features` is a sparse map from feature name to finite numeric value, or an
equivalent sequence of `(feature, value)` pairs. The detector does not require
row ids, labels, timestamps, a declared schema, categorical/numeric partitions,
or tuning parameters.

## Literature

### xStream

xStream is the closest direct match in the literature. It targets
feature-evolving streams, where both data points and the feature space evolve
over time. The paper represents stream elements as `(id, feature, delta)`
updates, which allows new feature names and feature-value changes without a
known dimensionality. It combines:

- StreamHash: sparse random projections keyed by feature name.
- Half-space chains: multi-scale density estimation over projected space.
- Count-min sketches: bounded-memory counts for bins.
- Windowed updates: adaptation to non-stationarity.

The KDD page describes xStream as constant-space and constant-time per incoming
update, using projections for high dimensionality and windowed updates for
non-stationarity. The paper also states that, among the compared methods, only
xStream supports evolving feature space and evolving feature values.

Useful sources:

- KDD 2018 page:
  <https://www.kdd.org/kdd2018/accepted-papers/view/xstream-outlier-detection-in-feature-evolving-data-streams>
- Paper PDF:
  <https://www.andrew.cmu.edu/user/lakoglu/pubs/18-kdd-xstream.pdf>
- Project page:
  <https://cmuxstream.github.io/>

Design implications:

- Strong basis for feature growth.
- Strong basis for sparse high-dimensional features.
- The original input contract is not a direct fit because it consumes `id` and
  delta updates. FeatureSketch instead uses a row-event contract that receives
  only the current feature map.
- Feature shrink is not a named first-class goal in the paper, but a sparse
  projection plus decayed/windowed counts can adapt when old features stop
  appearing.

### RS-Hash

RS-Hash is a randomized hashing detector for subspace outliers. IBM's summary
describes it as linear-time with constant space, using randomized hashing and
generalizable to data streams. It is simpler than xStream and relevant as a
baseline, but it assumes a more conventional fixed-row stream and does not solve
unknown feature growth as directly as xStream.

Useful source:

- IBM Research summary:
  <https://research.ibm.com/publications/subspace-outlier-detection-in-linear-time-with-randomized-hashing>

Design implications:

- Good baseline for high-dimensional subspace anomaly detection.
- Weaker fit for feature-evolving schemas because feature-name hashing and the
  projection layer would need to be added.
- Less expressive than half-space chains for multi-scale density.

### OAD-TDS

OAD-TDS is a newer method for trapezoidal data streams, where both instance and
feature space may expand. Its SSRN abstract describes dynamic feature weighting
for feature distribution changes and incremental locality-sensitive hashing for
instance state dynamics.

Useful source:

- SSRN page:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6030752>

Design implications:

- Relevant because it explicitly targets streams with feature expansion.
- Less mature as a design foundation than xStream: it is recent, and the public
  abstract emphasizes Dask/distributed scheduling rather than a compact
  in-process detector.
- The feature weighting idea is useful, but weights should remain internal to
  preserve the fixed public API.

## Recommendation

FeatureSketch should use a row-event adaptation of xStream, not a direct port
of the original triplet-update algorithm.

The detector accepts a single sparse feature map per event. Internally,
feature-name hashing keeps the model shape stable as new names appear. Sparse
event projection, presence-sensitive projections, and temporal decay handle
shrinking schemas. The resulting detector supports:

- feature evolving: new keys can appear at any time;
- feature shrink: missing keys do not cause dimension errors, and stale
  historical density fades out;
- feature-only input: no `id`, timestamp, label, or schema;
- fixed public behavior: sketch, projection, and decay constants stay internal.

This is a practical detector design rather than a paper-faithful xStream port.
The original xStream setting is more general because it maintains scores for
evolving object ids under delta updates. FeatureSketch narrows the contract to
scoring the next event from its currently observed features.

## Proposed Algorithm

The algorithm is named `FeatureSketch`: feature names define the input space,
and bounded sketches hold the evolving density model.

### Overview

```mermaid
flowchart TD
    A["Sparse feature event<br/>{feature_name: value}"] --> B["Normalize input<br/>combine duplicates<br/>asinh(value)"]
    B --> C["Feature-name hashing<br/>stable coefficient per<br/>(feature, projection)"]
    C --> D["Value projection<br/>numeric magnitude signal"]
    C --> E["Presence projection<br/>observed feature-set signal"]
    B --> F["Feature-count signal<br/>log1p(observed keys)"]
    D --> G["Value ensemble<br/>half-space chains"]
    E --> H["Presence ensemble<br/>half-space chains"]
    F --> H
    G --> I["Projected chain bins"]
    H --> I
    I --> J{"Method"}
    J -->|score| K["Read sketches<br/>compute anomaly score<br/>no mutation"]
    J -->|update| L["Apply lazy decay<br/>then increment sketch bins<br/>no scoring reads"]
    K --> M["Return computed score<br/>higher means more anomalous"]
    L --> N["Return no score"]
```

```mermaid
flowchart LR
    A["Feature evolving<br/>new key appears"] --> B["Hash key on demand"]
    B --> C["Projection shape unchanged"]
    C --> D["Sketch bins updated"]

    E["Feature shrink<br/>previous key disappears"] --> F["Presence projection changes"]
    F --> G["Score can rise immediately"]
    G --> H["Old bins decay over time"]
```

### Input normalization

For every event:

1. Accept sparse features as `(name, value)` pairs.
2. Reject non-finite values.
3. Combine duplicate feature names by summing their values, preserving the key
   even if the sum is exactly zero.
4. Apply `asinh(value)` before value projection so negative and large positive values
   are both supported.

The detector does not require a known feature universe. A dense vector can be
accepted by converting each index to a stable string key internally, but the
core representation should be sparse.

Absence is meaningful. A missing feature means the key is not part of the
current event and contributes to the presence signal by not appearing. A feature
whose combined value is exactly zero is still present: it contributes to the
presence projection, but contributes `asinh(0.0) = 0.0` to the value projection.
Categorical values should therefore be encoded as explicit one-hot feature names
only when the category is present, for example `status:401 -> 1.0`; boolean
false and missing categories should both omit the corresponding key unless the
application creates an explicit feature such as `flag:false -> 1.0`.

### Projection

Maintain `K_v` value projection dimensions and `K_p` presence projection
dimensions, chosen by internal constants. For each feature name `f` and
projected dimension `k`, derive a stable sparse random coefficient from the
detector seed and `(f, k)`:

```text
coef(f, k) in {-sqrt(3), 0, +sqrt(3)}
P(coef = 0) = 2/3
P(coef = +sqrt(3)) = 1/6
P(coef = -sqrt(3)) = 1/6
```

For each event, compute two projection vectors:

```text
value_projection[k] = sum(asinh(value_f) * coef(f, k))
presence_projection[k] = sum(coef(f, k)) for observed feature names
```

Also compute one scalar feature-count signal:

```text
feature_count_signal = log1p(number of observed feature names)
```

Use separate half-space chain ensembles for the value projection and the
presence vector. The presence vector is the presence projection with the
feature-count signal appended as one extra dimension:

```text
presence_vector = concat(presence_projection, [feature_count_signal])
```

The value ensemble detects unusual feature magnitudes. The presence ensemble
detects unusual feature sets, including feature shrink where a previously common
key disappears from an event.

The presence channel is the main adaptation beyond xStream. Without it, an
event that loses a key whose numeric value was usually small can look too close
to normal. Keeping presence in a separate ensemble prevents value-density bins
from hiding schema-change evidence. The scalar feature-count signal gives
feature shrink and expansion a direct low-dimensional path even when random
presence coefficients collide or cancel out.

### Density model

FeatureSketch uses two ensembles of half-space chains over the projected
vectors:

- each chain has fixed depth `D`;
- value-chain levels sample dimensions from `value_projection`;
- presence-chain levels sample dimensions from `presence_vector`, including the
  appended feature-count dimension;
- each level chooses its projected dimension and bin width from seeded constants;
- each level owns a count-min sketch for bin counts;
- `density_chain_level` is the minimum decayed count across the `R` rows of that
  level's count-min sketch for the selected bin;
- scoring uses the highest normalized surprise across levels, which is
  equivalent to the lowest clamped density ratio across levels;
- each chain level tracks a decayed reference mass for normalization, separate
  from the per-cell sketch counts;
- each ensemble converts low normalized densities into high anomaly
  contributions and averages those contributions.

The public anomaly score is higher for more anomalous events. Internally,
xStream-style density is lower for anomalies, so FeatureSketch exposes a
surprise score:

```text
density_ratio_chain_level = clamp(
    density_chain_level / max(reference_mass_chain_level, epsilon_mass),
    epsilon,
    1.0,
)
chain_surprise = max(-log(density_ratio_chain_level)) across chain levels
ensemble_surprise = mean(chain_surprise) across chains
```

where `epsilon` prevents `log(0)` and `epsilon_mass` only prevents division by
zero. This is not a calibrated probability; it is a volume-normalized surprise
score. Normalizing by the decayed reference mass of the same chain level keeps
the score scale more stable across warm streams, traffic bursts, and
long-running decay than a raw reciprocal density. The upper clamp at `1.0`
means common or over-dense bins contribute zero surprise, while rare bins
contribute positive surprise.

The final score is the average of the value-ensemble surprise and the
presence-ensemble surprise:

```text
score = mean(value_surprise, presence_surprise)
```

### Online update order

For `score(features)`, compute the score against the current reference state
without mutation.

For `update(features)`, perform only the update. It still computes projections
and chain-level bin assignments because the sketch update locations are defined
in projected space, but it skips the scoring reads and anomaly-score reduction.

When a caller needs both operations for the same event, call `score(features)`
before `update(features)`. Scoring after update is also valid if the desired
meaning is "how anomalous is this event after it has been incorporated," but
the default online-detection pattern is score-before-update.

### Adaptation and shrink handling

FeatureSketch uses lazy exponential decay. It adapts continuously without
exposing a window boundary in the public API:

- every update increments an internal event counter;
- each sketch cell stores `(count, last_seen_epoch)`;
- each chain level stores `(reference_mass, last_seen_epoch)` for the same lazy
  decay schedule used by sketch cells;
- reading or writing a cell applies lazy decay based on elapsed events;
- old features and old bins naturally lose influence without explicit deletion.

This handles global feature shrink: if a feature disappears from the stream,
its historical bins stop receiving updates and decay away. It handles per-event
feature shrink through the presence projection because the event's observed key
set changes even if values are otherwise normal.

FeatureSketch intentionally does not special-case cold start. Early scores are
unstable because the density sketches have not yet accumulated a useful
reference distribution. As operational guidance, production pipelines can ignore
or down-rank roughly the first internal half-life of scores when startup
behavior matters. This is not a readiness invariant; it is a simple default
warmup policy.

The detector should not maintain a dense registry of all feature names. A small
optional diagnostic sketch for feature frequencies is acceptable, but the core
algorithm should remain bounded by the projection/chains/sketch constants, not
by the number of feature names ever seen.

## Fixed Internal Defaults

These values are fixed implementation constants rather than builder parameters
in the first public version:

| Internal constant              |    Suggested value | Reason                                                            |
| ------------------------------ | -----------------: | ----------------------------------------------------------------- |
| Value projection dimensions    |                 32 | Enough for sparse random projection while keeping update cost low |
| Presence projection dimensions |                 32 | Keeps schema-change detection independent from value magnitudes   |
| Chains per ensemble            |                 16 | Ensemble stability without large memory                           |
| Chain depth                    |                  8 | Multi-scale bins without excessive sketch reads                   |
| Sketch rows                    |                  2 | Same practical shape as current `MStream` defaults                |
| Sketch buckets                 |               2048 | More room than `MStream` because projected bins are more varied   |
| Decay half-life                |        2048 events | Tracks recent behavior while preserving a useful baseline         |
| Epsilon                        |              1e-12 | Prevents `log(0)` in density-ratio scoring                        |
| Epsilon mass                   |              1e-12 | Prevents division by zero before enough mass has accumulated      |
| Seed                           | fixed library seed | Deterministic behavior without a public option                    |

These are implementation constants, not public configuration. Test-only
constructors may inject a seed internally, but the public detector should not
expose tuning knobs in the first version.

## Complexity

Let:

- `m` be the number of present feature names in the current event, including
  zero-valued present features;
- `K_v` be the number of value projection dimensions;
- `K_p` be the number of presence projection dimensions;
- `C` be chains per ensemble;
- `D` be chain depth;
- `R` be sketch rows;
- `B` be sketch buckets.

FeatureSketch does not store the historical feature universe, so long-run memory
does not grow with the number of distinct feature names ever observed.

### Time per event

Input normalization is `O(m)`, assuming feature names can be hashed in time
proportional to their byte length. Projection computes one value coefficient and
one presence coefficient for each present feature and projected dimension:

```text
O(m * (K_v + K_p))
```

Scoring reads sketch cells for every chain level in both ensembles. With
count-min sketches, each level reads `R` cells and uses the minimum decayed count
as that level's density estimate:

```text
O(2 * C * D * R)
```

Committed updates reuse the projected chain bins, write the same number of
sketch cells, and update the same per-level reference masses:

```text
O(2 * C * D * R)
```

Therefore:

```text
score(features):            O(m * (K_v + K_p) + 2 * C * D * R)
update(features):           O(m * (K_v + K_p) + 2 * C * D * R)
```

Calling `score(features)` followed by `update(features)` performs both the
scoring reads and committed-update writes, so it costs
`O(m * (K_v + K_p) + 4 * C * D * R)` in a straightforward implementation. With
the fixed defaults, the sketch part is constant per event. Runtime is linear in
the number of present features in the event and independent of the number of
feature names seen historically.

### Space

The persistent sketch storage is:

```text
O(2 * C * D * R * B)
```

The factor `2` is for the value and presence ensembles. Per-level reference
masses add `O(2 * C * D)` state, which is dominated by sketch cells.

Per-event temporary storage materializes the normalized event, projection
vectors, and chain-level bin assignments:

```text
O(K_v + K_p + m + 2 * C * D)
```

where `m` covers the normalized event map when duplicate feature names must be
combined. An implementation can recompute bin assignments instead of storing
them, reducing temporary storage to `O(K_v + K_p + m)` at the cost of extra
hash/binning work.

## API Sketch

Rust:

```rust
use rcf3::FeatureSketch;

let mut detector = FeatureSketch::new();

let event = [
    ("endpoint:/login", 1.0),
    ("status:401", 1.0),
    ("bytes", 812.0),
];
let score = detector.score(event)?;
detector.update(event)?;

let preview = detector.score([
    ("endpoint:/admin", 1.0),
    ("status:401", 1.0),
    ("bytes", 12000.0),
])?;
```

Python:

```python
from rcf3 import FeatureSketch

detector = FeatureSketch()
event = {
    "endpoint:/login": 1.0,
    "status:401": 1.0,
    "bytes": 812.0,
}
score = detector.score(event)
detector.update(event)
```

The categorical/numeric split is intentionally absent. Categorical features are
represented by one-hot style feature names with value `1.0`; numeric features
use their natural finite values. One-hot encoders should omit inactive
categories rather than emitting inactive keys with value `0.0`, because a
zero-valued key is still treated as present.

## Remaining Design Decisions

1. Dense input support: accepting `&[f32]` is convenient, but sparse named
   features are the better primary contract for evolving/shrinking schemas.
2. Serialization: include the fixed constants in serialized state so future
   versions can reject incompatible states cleanly.

## Validation Plan

Minimum regression scenarios:

1. Feature growth: using a deterministic internal seed, warm for at least one
   internal half-life on `{a, b}`, collect a small baseline of normal `{a, b}`
   scores, then assert `{a, b, new_feature}` scores above the baseline median
   before adaptation.
2. Feature shrink: using a deterministic internal seed, warm for at least one
   internal half-life on `{a, b, c}`, collect a small baseline of normal
   `{a, b, c}` scores, then assert `{a, b}` scores above the baseline median
   before adaptation.
3. Shrink adaptation: after many `{a, b}` updates, assert `{a, b}` no longer
   remains permanently anomalous.
4. Sparse high cardinality: stream many unique feature names and assert memory
   remains bounded by sketch sizes.
5. Score purity: `score(x)` should not mutate detector state; scoring the same
   event twice from the same state should return the same value.
6. Duplicate names: duplicate feature entries should match a pre-combined map.
7. Zero and absence: a feature whose combined value is exactly zero should still
   affect the presence projection and should not match omitting that feature from
   the same event.
8. Signed values: positive and negative finite values are accepted; NaN and
   infinity are rejected.

## Conclusion

FeatureSketch is an xStream-inspired sparse projection detector with an
explicit presence projection and internal temporal decay. It is a better fit for
schema-evolving streams than adapting `Forest`, `OnlineIForest`, or `MStream`:
those detectors require fixed dimensions or separate feature categories, while
FeatureSketch lets the schema grow and shrink without a public schema or tuning
surface.
