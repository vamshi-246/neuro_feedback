#!/usr/bin/env bash
# RTL simulation with Icarus Verilog. Run from the repo root:
#     bash sim/run_sim.sh            # filter unit test + all 10 system epochs
#     bash sim/run_sim.sh unit       # filter engine only
#     bash sim/run_sim.sh top 3      # system test, first 3 epochs only
#
# Vectors (unit_*.mem, ep_raw.mem, exp_*.mem) are committed, so this needs no EEG
# data. Regenerate them from the dataset with:
#     py scripts/hardware/hw_golden.py --write-mem
#     py scripts/hardware/gen_filter_unit.py sim
#     py scripts/hardware/gen_test_vectors.py --out sim
#
# Icarus is slow on this design (~2.5k cycles/s: 48-88 bit arithmetic). One epoch is
# ~2.4 M cycles, about 15-20 minutes, so epochs are run as parallel processes.
set -e
cd "$(dirname "$0")"
cp ../RTL/mem/*.mem .
what=${1:-all}
nep=${2:-10}

if [ "$what" = "unit" ] || [ "$what" = "all" ]; then
    iverilog -g2005 -o tb_ff.vvp ../Testbench/tb_filtfilt_engine.v \
        ../RTL/filtfilt_engine.v ../RTL/rom_sync.v ../RTL/sdp_ram.v
    for b in 0 1 2 3 4 5; do vvp -n tb_ff.vvp +B0=$b +NB=1 > ff_band$b.log 2>&1 & done
    wait
    cat ff_band*.log
fi

if [ "$what" = "top" ] || [ "$what" = "all" ]; then
    iverilog -g2005 -o tb_top.vvp ../Testbench/tb_pain_classifier_top.v ../RTL/*.v
    for ((e = 0; e < nep; e++)); do vvp -n tb_top.vvp +E0=$e +NEP=1 > top_ep$e.log 2>&1 & done
    wait
    grep -h "epoch\|PASS\|FAIL" top_ep*.log
fi
