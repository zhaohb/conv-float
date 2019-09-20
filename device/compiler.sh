#!/bin/bash

aoc -march=emulator -report -v -g Conv2D.cl
env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 gdb --args ./host 
