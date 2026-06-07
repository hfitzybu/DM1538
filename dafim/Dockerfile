# Ubuntu-based build/runtime image for DAFIM (MPI)
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openmpi-bin libopenmpi-dev \
    libmetis-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN make

# Default command prints CLI usage.
CMD ["bash", "-lc", "./bin/dafim_mpi --help || true"]
