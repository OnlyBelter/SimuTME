# Parallel Mixed candidate generation

Add an opt-in CPU multiprocessing path for the expensive Mixed candidate
generation loop. Preserve the scientific configuration and existing serial
behavior. Measure speedup on the target server before claiming a 10-fold gain.

## Evidence and goal

The `current_best_2` N100K log records 1,673 complete 5,000-candidate batches.
Of 8,365,000 candidates, 7,957 pass the mGEP filter, an acceptance rate of
0.0951%. Fraction-prior rejection accepts approximately 9.14% of its
proposals. Average batch time is 8.08 seconds: 1.45 seconds for fractions,
0.556 for SCT sampling, 4.91 for aggregation, 1.06 for filtering, and 0.102
for saving and metadata.

At this observed rate, candidate generation for 100,000 accepted profiles
projects to approximately 47.2 hours. This is an extrapolation, not a
completed run or an end-to-end measurement. Aggregation accounts for about
61% of batch time; eliminating it alone would yield at most about 2.55-fold
loop speedup. Concurrent complete batches can use the server's CPU capacity
across all stages, subject to memory bandwidth and other workloads.

## Scope and configuration

Target eager-loaded SCT references, segment fraction sampling, random SCT
selection, and distance-filtered Mixed generation with `mgep`. Preserve
existing serial paths for other configurations. The first version does not
change the sampling distribution, GEP filter, reference, selected genes,
candidate batch size, or output data format. GPU support is outside this
change and may be evaluated after CPU benchmarks.

Add dataset-overridable settings:

- `mixed_candidate_workers: 0` by default retains the existing serial path.
  A positive integer opts into the parallel engine; `1` runs the same engine
  with one worker for reproducibility and benchmarking.
- `mixed_candidate_seed` is a required nonnegative integer when the parallel
  engine is enabled. Reject booleans, invalid values, and unsupported
  combinations before starting generation.

Place both settings in `src/config_example.yaml` under `parameters`, with
comments explaining their defaults. Values in a Mixed dataset override global
values. Omitted keys use the same defaults as the example:

```yaml
parameters:
  mixed_candidate_workers: 0  # 0 retains the existing serial engine
  mixed_candidate_seed: null  # Required only when workers is positive
```

For a server run, enable the engine only for a supported target dataset:

```yaml
datasets:
  Mixed_N100K:
    mixed_candidate_workers: 8  # Starting point; benchmark on the server
    mixed_candidate_seed: 42  # Reproducible parallel batch streams
```

An omitted or zero worker count retains the current serial engine, including
its existing loading-mode behavior and randomness; it does not require a
seed. Positive worker counts require an eager-loaded SCT reference. A local
computer can keep the serial defaults, including its existing `auto` or
`lazy` SCT loading mode. Do not silently switch a parallel request to serial
when the SCT loading mode resolves to `lazy`; report the conflict before
generation. Avoid extra tuning parameters until benchmarks justify them.

Do not modify the user's existing YAML or running dataset. Use a new dataset
name and output location for a parallel run. An existing serial intermediate
cannot be adopted as a parallel checkpoint.

## Execution

The parent resolves the selected genes, reference distances and cutoffs,
sampling metadata, and eager SCT expression once. It starts persistent
workers that use one read-only shared expression matrix, rather than loading
a separate copy per worker. Use a platform-appropriate shared-memory or
memory-mapped mechanism and validate storage capacity before dispatch.
Clean up owned resources after normal completion, worker errors, and
interruptions; never unlink source data or pre-existing outputs.

The parent assigns monotonically increasing batch IDs. Each worker creates
the complete batch: prior-constrained fractions, random SCT selection,
full-gene aggregation and normalization, and mGEP filtering. Keep the
existing floating-point precision, rounding, coefficient handling, and
normalization order. In particular, do not discard non-filtering genes
before full-gene normalization. Workers return only accepted expression,
fractions, selected IDs, and counts; the parent alone writes outputs and
filtering metadata.

Bound in-flight batches and result buffering to a small multiple of worker
count. Consume results in batch-ID order regardless of completion order,
count all candidates in committed batches, and trim only the final accepted
batch to the requested output count. Discard uncommitted speculative results
after the target is met. Limit internal numeric-library threads per worker to
avoid oversubscription, and log worker count, batch size, and elapsed times.

## Randomness and recovery

Derive independent deterministic random streams from the master seed and
batch ID for every random operation in the supported path. Do not let forked
workers inherit identical streams. With the same seed, inputs, software, and
batch size, committed results must be independent of worker count and
completion order. Parallel output is not expected to match historical serial
draws sample-for-sample.

Record the engine version, seed, and batch size in cache identity and record
worker count as execution metadata. A changed seed or engine identity cannot
reuse an incompatible cache. Worker count alone does not change batch
identity or results.

The coordinator records a durable committed-batch checkpoint, including
the next batch ID, accepted count, and filter counters, even when a batch
accepts no profiles. Resume only when the checkpoint and all existing
intermediate files are mutually consistent. A worker failure stops the run
and reports the batch ID. If interruption leaves a partial multi-file write,
fail clearly without silently deleting, truncating, or overwriting artifacts.
Do not reuse a checkpoint when its scientific or random identity differs.
An exclusive per-target lock prevents concurrent parallel coordinators from
writing the same files. A forced termination can leave a stale lock; require
the user to verify that the original process is gone before removing it.

## Validation

Replay identical fractions and selected SCT IDs through serial and worker
calculations. Check expression, selected gene order, mGEP distances, inclusive
cutoffs, filtering masks, labels, and output precision. Test prior bounds,
zero survivors, final-batch trimming, and one-worker versus multi-worker
results. Verify unique batch seeds, worker-count independence, clean resource
shutdown, default serial behavior, global and dataset-level configuration
precedence, and clear errors for unsupported settings.

Test a completed checkpoint resume, including batches with no survivors,
worker failure, inconsistent or partial files, and mismatched cache identity.
Check that no two processes write shared output files. Run the full existing
SimuTME test suite and retain the serial default.

## Benchmark

Provide an independent, reproducible benchmark that includes fraction
generation, SCT sampling, aggregation, mGEP filtering, and representative
parent-side writes. Compare the existing serial path against the new engine
with 1, 4, 8, and 16 workers, using the same input and scientific settings.
Use fixed candidate batches, warm-up, repeated measurements, and a fresh
temporary output location.

Report setup and steady-state time separately, candidate and accepted
profiles per second, rejection rates, worker count, numeric-library threads,
software and hardware details, total process-tree peak memory, and output
integrity. Report variation and any omitted finalization, PCA, or storage
costs. Run a representative bounded workload with the actual SCT reference
on the user's server; do not present local or synthetic timing as a server
result. A 10-fold gain is an aspiration, not an acceptance guarantee.

## Documentation

Document configuration defaults, supported combinations, seeding,
checkpoint restrictions, memory requirements, benchmark commands, and the
fact that parallel draws differ from prior serial draws. Update the
changelog when the implementation is complete.
