# Dataset-local parallel temporary storage

Parallel Mixed simulation stores a memory-mapped SCT expression matrix and,
when subgroup PCA is active, a reference matrix in the system temporary
directory. Move these files into a unique temporary subdirectory of the
dataset's output directory, so they use the output filesystem instead of
`/tmp` by default.

## Storage and cleanup

For each parallel candidate run, create a unique directory with the existing
`simutme-candidates-` prefix under the relevant Mixed dataset result directory.
Use the `simulation_for_stat` result directory for statistics-only runs.
Place both the SCT and optional GEP reference matrices in that directory.
Existing serial behavior is unchanged.

Keep the current temporary-directory lifecycle: close worker processes before
removing the directory, and remove it on normal completion or a handled
exception. A forced kill, machine failure, or filesystem error can leave the
directory behind. Don't automatically remove directories from earlier runs;
they may belong to active processes. Check that no run is using a leftover
directory before manually removing it.

Check free space on the selected output filesystem before writing each matrix.
Report the selected directory and the required and available space when the
check fails. Don't silently fall back to `/tmp` or remove existing outputs.
The output filesystem needs space for the temporary matrices as well as
simulation outputs.

## Compatibility

Don't change candidate generation, seeds, filtering, scheduling, checkpoints,
or output identities. A compatible interrupted simulation can resume using a
new temporary directory. Don't move files used by an already running process.

## Verification

Test full and statistics-only parallel runs, both with and without a subgroup
reference. Verify that temporary files are created under the appropriate
dataset output directory and removed after normal completion and handled
errors. Verify that insufficient space reports the output location, and that
existing parallel determinism and resume tests continue to pass.

Update the configuration and Mixed workflow documentation to describe the new
location, storage requirements, and cleanup limitations.
