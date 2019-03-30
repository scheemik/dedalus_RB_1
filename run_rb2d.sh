#!/bin/bash
# A bash script to run the 2D Rayleigh Benard python code

RA=1.1e6
PR=1.1

echo "Removing frames, snapshots, and RB.gif"
#rm -r frames
#rm -r snapshots
#rm RB.gif
echo "Running Rayleigh Benard script"
#python3 rayleigh_benard.py
#python3 rb_with_S.py $RA $PR
echo ""
echo "Merging snapshots"
#python3 merge.py snapshots
echo ""
echo "Plotting 2d series"
python3 plot_2d_series.py snapshots/*.h5 --rayleigh=$RA --prandtl=$PR
echo ""
echo "Creating gif"
python3 create_gif.py
echo "Done"

