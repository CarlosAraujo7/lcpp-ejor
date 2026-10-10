#!/bin/bash
export JULIA_NUM_THREADS=$1
julia --check-bounds=no runPCGLNS.jl "${@:2}"