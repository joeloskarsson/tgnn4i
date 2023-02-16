#!/bin/sh

# Traffic
for ds in bay_node_0.25 bay_node_0.5 bay_node_0.75 bay_node_1.0 la_node_0.25 la_node_0.5 la_node_0.75 la_node_1.0
do
    python predict_prev.py --dataset $ds --test 1
done

# USHCN
for ds in ushcn_tmin_10nn ushcn_tmax_10nn
do
    python predict_prev.py --dataset $ds --test 1
done

