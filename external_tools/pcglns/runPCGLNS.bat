@echo off
set JULIA_NUM_THREADS=%1
for /f "tokens=1,* delims= " %%a in ("%*") do set ALL_BUT_FIRST=%%b
julia --check-bounds=no runPCGLNS.jl %ALL_BUT_FIRST%