#!/bin/bash
# A bash script to run the 2D Rayleigh Benard python code

echo "running Rayleigh Benard script"
python3 rayleigh_benard.py
echo ""
echo "Merging snapshots"
python3 merge.py snapshots
echo ""
echo "Plotting 2d series"
python3 plot_2d_series.py
echo ""
echo "Creating gif"
python3 create_gif.py