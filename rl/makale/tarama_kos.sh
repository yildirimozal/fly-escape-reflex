#!/bin/zsh
# Bedel taraması: her B için 3 tohum paralel (sırayla B'ler), ardından refleks ve hiç-kalkmayan.
cd "$(dirname "$0")"
P=${PYTHON:-python}
for B in 10 30 60 100; do
  for s in 1 2 3; do $P -u tarama.py ogren $s 100000 $B > tarama/ogren_b${B}_s${s}.log 2>&1 & done
  wait
  echo "B=$B bitti $(date +%H:%M)"
done
for B in 10 30 60 100; do
  $P -u tarama.py refleks 300 $B > tarama/refleks_b${B}.log 2>&1 &
  $P -u tarama.py hic 300 $B > tarama/hic_b${B}.log 2>&1 &
  wait
done
echo "TARAMA BITTI $(date +%H:%M)"
