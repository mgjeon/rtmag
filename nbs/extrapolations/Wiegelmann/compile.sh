#! /bin/bash
gcc -o prepro NLFFF/prepro/prepro.c -lm

cd NLFFF
make
rm -rf *.o
mv relax1 ../relax1