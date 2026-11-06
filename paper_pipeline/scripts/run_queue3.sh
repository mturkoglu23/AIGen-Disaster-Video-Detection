#!/usr/bin/env bash
# Üçüncü kuyruk: TriStream-V2 (özellik uzayında zamansal akış + koşullu meta veri adversaryı) ve ablasyonları
cd "$(dirname "$0")/.."
PY=${PY:-python}
run() { tag=$1; shift
  if [ -f "results/runs/$tag/summary.csv" ]; then echo "skip $tag"; return; fi
  rm -rf "results/runs/$tag"; echo "$(date) START $tag" >> results/queue.log
  $PY train.py --tag "$tag" "$@" > "results/stdout_$tag.txt" 2>&1
  echo "$(date) END $tag rc=$?" >> results/queue.log; }
V2="--model tristream2 --data clean_pairs --split group --recipe proposed --robust"
run V_v2_full          $V2 --streams sem,ftmp,for --adv_w 0.3
run V_v2_sem_ftmp      $V2 --streams sem,ftmp     --adv_w 0.3
run Ab2_full_noadv     $V2 --streams sem,ftmp,for --adv_w 0
run Ab2_sem_ftmp_noadv $V2 --streams sem,ftmp     --adv_w 0
run Ab2_sem_adv        $V2 --streams sem          --adv_w 0.3
run Ab2_ftmp           $V2 --streams ftmp         --adv_w 0
run L_v2_full --model tristream2 --data clean_pairs --split lodo --recipe proposed --streams sem,ftmp,for --adv_w 0.3
echo "$(date) QUEUE3 DONE" >> results/queue.log
