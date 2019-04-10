#!/bin/bash
# A bash script to run the 2D Rayleigh Benard python code

# which version? 1=b, 2=T, 3=S
VER=1
# Define parameters
RA=1e5
PR=7
TAU=1.1e-2
RP=1e-2

echo "Removing frames, snapshots, and RB.gif"
rm -r frames
rm -r snapshots

if [ $VER -eq 1 ]
then
  rm b_RB.gif
  echo "Running Rayleigh Benard script"
  python3 b_rb_code.py $RA $PR
  echo ""
  echo "Merging snapshots"
  python3 merge.py snapshots
  echo ""
  echo "Plotting 2d series"
  python3 b_plot_2d_series.py snapshots/*.h5 --rayleigh=$RA --prandtl=$PR
  echo ""
  echo "Creating gif"
  python3 create_gif.py b_RB.gif
elif [ $VER -eq 2 ]
then
  rm T_RB.gif
  echo "Running Rayleigh Benard script"
  python3 T_rb_code.py $PR $TAU $RP
  echo ""
  echo "Merging snapshots"
  python3 merge.py snapshots
  echo ""
  echo "Plotting 2d series"
  python3 T_plot_2d_series.py snapshots/*.h5 --prandtl=$PR --diffusivity_r=$TAU --density_r=$RP
  echo ""
  echo "Creating gif"
  python3 create_gif.py T_RB.gif
else # VER = 3
  rm S_RB.gif
  echo "Running Rayleigh Benard script"
  python3 S_rb_code.py $PR $TAU $RP
  echo ""
  echo "Merging snapshots"
  python3 merge.py snapshots
  echo ""
  echo "Plotting 2d series"
  python3 S_plot_2d_series.py snapshots/*.h5 --prandtl=$PR --diffusivity_r=$TAU --density_r=$RP
  echo ""
  echo "Creating gif"
  python3 create_gif.py S_RB.gif
fi

echo "Done"
