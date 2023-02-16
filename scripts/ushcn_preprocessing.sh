#!/bin/sh

for target in tmin tmax
do
    python preprocess_ushcn.py --target $target
done

