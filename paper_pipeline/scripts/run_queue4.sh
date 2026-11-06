#!/usr/bin/env bash
# Dördüncü kuyruk: çift omurgalı TriStream (CLIP-L + DINOv2-B anlamsal akışları + izsel + zamansal)
cd "$(dirname "$0")/.."
PY=${PY:-python}
run() { tag=$1; shift
  if [ -f "results/runs/$tag/summary.csv" ]; then echo "skip $tag"; return; fi
  rm -rf "results/runs/$tag"; echo "$(date) START $tag" >> results/queue.log
  $PY train.py --tag "$tag" "$@" > "results/stdout_$tag.txt" 2>&1
  echo "$(date) END $tag rc=$?" >> results/queue.log; }
DUAL="--model tristream --data clean_pairs --recipe proposed --sem_backbone vit_large_patch14_clip_224.openai"
run P_dual            $DUAL --split group --streams sem,sem2,for,tmp --robust
run Ab_dual_sem_sem2  $DUAL --split group --streams sem,sem2 --robust
run L_dual            $DUAL --split lodo  --streams sem,sem2,for,tmp
run P_dual_s1         $DUAL --split group --streams sem,sem2,for,tmp --robust --seed 1
run P_dual_s2         $DUAL --split group --streams sem,sem2,for,tmp --robust --seed 2
echo "$(date) QUEUE4 DONE" >> results/queue.log
