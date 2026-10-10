# external_tools

Third-party code and data needed to run the reference solver used in the comparison. Nothing here is part of
the contribution of this project.

## `pcglns/` — PCGLNS (Kuryavtsev & Khachay)

Source: <https://github.com/AndreiKud/PCGLNS> (Apache-2.0, see `pcglns/LICENSE`). Only the essentials were
copied. Written in Julia with no external dependencies (`Project.toml` declares only stdlib packages).

**Requires Julia 1.6.** The current default release (1.12) crashes on this code with
`GC error (probable corruption)`. Install with `juliaup add 1.6` and run

```bash
julia +1.6 --check-bounds=no runPCGLNS.jl <instance.pcglns> -max_time=N -output=<file>
```

The Python wrapper `pcgtsp_comparison/run_comparison.py` does this for you; the Julia executable, the juliaup
channel and the scratch directory can be set with `LCPP_JULIA`, `LCPP_JULIA_CHANNEL` and `LCPP_TMP_DIR`
(see the top-level README).

`PCGTSPLIB/` and `PCGLNS_PCGTSP/` hold **33 of the 39 instances** of the PCGTSPLIB (Salman, Ekstedt & Damaschke,
2020): the ESC\*, br17, ft53, ft70, p43, prob.42/100, kro124p, rbg048a/050c/109a and ry48p families. `.pcgtsp`
is the original format; `.pcglns` is already converted to the solver's input format (this is what is run).

Left out to keep the repository small: `rbg150a` onwards (each 4-21 MB of full distance matrices) and the generic
GTSPLIB / SOPLIB libraries the PCGTSPLIB was derived from (~140 MB). They can be obtained with
`git clone --depth 1 https://github.com/AndreiKud/PCGLNS.git`.
