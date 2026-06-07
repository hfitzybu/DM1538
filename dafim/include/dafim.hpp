#pragma once

// Public entrypoint.
// MPI must be initialized before calling.

extern int RANK;
extern int SIZE;

int dafim_run(int argc, char** argv);
