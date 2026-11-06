#!/usr/bin/env bash
# Beşinci kuyruk: ana model CLIP anlamsal + zamansal için LODO ve tohum tekrarları
cd "$(dirname "$0")/.."
PY=${PY:-python}
run() { tag=$1; shift
  if [ -f "results/runs/$tag/summary.csv" ]; then echo "skip $tag"; return; fi
  rm -rf "results/runs/$tag"; echo "$(date) START $tag" >> results/queue.log
  $PY train.py --tag "$tag" "$@" > "results/stdout_$tag.txt" 2>&1
  echo "$(date) END $tag rc=$?" >> results/queue.log; }
C="--model tristream --data clean_pairs --recipe proposed --sem_backbone vit_large_patch14_clip_224.openai"
run L_clip_semtmp      $C --split lodo  --streams sem,tmp
run P_clip_semtmp_s1   $C --split group --streams sem,tmp --robust --seed 1
run P_clip_semtmp_s2   $C --split group --streams sem,tmp --robust --seed 2
echo "$(date) QUEUE5 DONE" >> results/queue.log
