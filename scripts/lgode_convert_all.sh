#!/bin/sh

# Traffic
for ds in bay la
do
    for on in 1.0 0.75 0.5 0.25
    do
        python convert_to_lgode.py --dataset $ds\_node\_$on
    done
done

# Periodic
python convert_to_lgode.py --dataset periodic_20_0.5_42

