# DAFIM (MPI) — Distributed Attribute-Aware Fair Influence Maximization

This repo is a cleaned-up, open-source-friendly packaging of the original single-file prototype. **Core computation logic is unchanged**; only the project layout and console output are streamlined.

## What this version outputs
For each seed ratio (1% ... 5%), it prints **only**:
- `Baseline`: `spread` and `minmax` (Min-Max ratio)
- `DAFIM`: `spread` and `minmax` (Min-Max ratio)

Example output:
```
ratio=0.010
[Baseline] spread=123.45, MinMax=0.8123
[DAFIM]    spread=120.10, MinMax=0.9011dafim_core
```

## Build
### Dependencies
- MPI (OpenMPI / MPICH)
- METIS (libmetis)

#### Ubuntu / Debian quick install
```bash
sudo apt-get update
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev libmetis-dev
```

#### macOS (Homebrew) quick install
```bash
brew install open-mpi metis
```

> If you see `fatal error: 'mpi.h' file not found`, it usually means you're compiling with `clang++`
> instead of the MPI wrapper. Use:
> ```bash
> make MPICXX=mpicxx
> ```

### Compile
```bash
make
```

If METIS is installed in a non-standard location:
```bash
make METIS_INC=/path/to/metis/include METIS_LIB=/path/to/metis/lib
```

The binary will be produced at:
- `bin/dafim_mpi`

## Run
```bash
mpirun -np 8 ./bin/dafim_mpi \
  --edge <REL.txt> --attr <ATT.txt> \
  --metis-parts 32 \
  --p-default 0.01 \
  --ic-trials 500 \
  --scheme inv \
  --lambda-homo 2.0
```

### (Optional) Docker on Ubuntu
If you want a fully reproducible Ubuntu build/run environment:
```bash
docker build -t dafim_mpi:ubuntu .
docker run --rm -it dafim_mpi:ubuntu
```

To run on your own input files, mount a data folder:
```bash
docker run --rm -it -v "$PWD/data":/data dafim_mpi:ubuntu \
  bash -lc "mpirun -np 8 ./bin/dafim_mpi --edge /data/REL.txt --attr /data/ATT.txt --metis-parts 32 --p-default 0.01 --ic-trials 500 --scheme inv --lambda-homo 2.0"
```

### Arguments (same semantics as original code)
- `--edge`: edge list file
- `--attr`: node attribute file
- `--metis-parts`: number of partitions (>= #MPI ranks)
- `--p-default`: IC probability
- `--ic-trials`: Monte Carlo trials
- `--seed`: random seed
- `--use-ghost`: enable ghost nodes in stage-1 RR sampling
- `--scheme`: AAFER reweight scheme (`inv` / `exp`)
- `--alpha`: smoothing constant used in AAFER
- `--theta2`: RR-set budget for stage-2 (`<=0` uses the original heuristic)
- `--lambda-homo`: homophily penalty strength for DAFIM weighting

## Project layout
- `src/main.cpp` — MPI bootstrap (init/finalize)
- `src/dafim_core.cpp` — original core logic (kept as a single translation unit for correctness)
- `include/dafim.hpp` — public entrypoint declaration

## Notes
- The method formerly labeled `AAFER-hybrid` is renamed to **DAFIM** in the output.
- All other console prints from the prototype were removed to keep the output clean and script-friendly.

## License
Add your preferred license (e.g., MIT) in `LICENSE`.
