#!/usr/bin/env bash
# Altıncı kuyruk: CLIP TriStream için eksik ablasyonlar (ikili kombinasyonlar + tekli kolların dayanıklılık testi)
cd "$(dirname "$0")/.."
PY=${PY:-python}
run() { tag=$1; shift
  if [ -f "results/runs/$tag/summary.csv" ]; then echo "skip $tag"; return; fi
  rm -rf "results/runs/$tag"; echo "$(date) START $tag" >> results/queue.log
  $PY train.py --tag "$tag" "$@" > "results/stdout_$tag.txt" 2>&1
  echo "$(date) END $tag rc=$?" >> results/queue.log; }
C="--model tristream --data clean_pairs --split group --recipe proposed --sem_backbone vit_large_patch14_clip_224.openai --robust"
run Ab_clip_sem_for_R $C --streams sem,for
run Ab_clip_sem_tmp_R $C --streams sem,tmp
run Ab_for_R          $C --streams for
run Ab_tmp_R          $C --streams tmp
run Ab_for_tmp_R      $C --streams for,tmp
echo "$(date) QUEUE6 DONE" >> results/queue.log
