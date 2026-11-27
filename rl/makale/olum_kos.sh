#!/bin/zsh
# Ölüm bedeli düzeltilmiş öğrenen devre: K=100 için her B'de 3 tohum paralel; ardından K=10 duyarlılığı (B=100).
# Beyin modeli taraması (kod/seker_esik.py) bitmeden başlamaz: aynı anda tek ağır iş.
cd "$(dirname "$0")"
P=${PYTHON:-python}
while pgrep -f "seker_esik.py tara" > /dev/null; do sleep 60; done
echo "basladi $(date +%H:%M)"
for KB in "100 10" "100 30" "100 60" "100 100" "10 100"; do
  K=${KB% *}; B=${KB#* }
  for s in 1 2 3; do $P -u olum.py ogren $s 100000 $B $K > tarama/olum_k${K}_b${B}_s${s}.log 2>&1 & done
  wait
  echo "K=$K B=$B bitti $(date +%H:%M)"
done
echo "OLUM BITTI $(date +%H:%M)"
