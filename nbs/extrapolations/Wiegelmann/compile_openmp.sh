#! /bin/bash
gcc -o prepro NLFFF/prepro/prepro.c -lm

cd NLFFF
make -f Makefile_openmp
rm -rf *.o
mv relax4 ../relax1_openmp