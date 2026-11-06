#!/usr/bin/env bash
# İkinci kuyruk: CLIP-L semantik omurgalı önerilen model + tohum tekrarları
cd "$(dirname "$0")/.."
PY=${PY:-python}
run() { tag=$1; shift
  if [ -f "results/runs/$tag/summary.csv" ]; then echo "skip $tag"; return; fi
  rm -rf "results/runs/$tag"; echo "$(date) START $tag" >> results/queue.log
  $PY train.py --tag "$tag" "$@" > "results/stdout_$tag.txt" 2>&1
  echo "$(date) END $tag rc=$?" >> results/queue.log; }
# ablasyonların dayanıklılık testleri (ilk ablasyon koşuları --robust olmadan yapılmıştı)
run Ab_sem_R      --model tristream --data clean_pairs --split group --recipe proposed --streams sem --robust
run Ab_sem_tmp_R  --model tristream --data clean_pairs --split group --recipe proposed --streams sem,tmp --robust
run Ab_sem_for_R  --model tristream --data clean_pairs --split group --recipe proposed --streams sem,for --robust
run P_tristream_clip --model tristream --data clean_pairs --split group --recipe proposed --sem_backbone vit_large_patch14_clip_224.openai --robust
run Ab_sem_clip      --model tristream --data clean_pairs --split group --recipe proposed --sem_backbone vit_large_patch14_clip_224.openai --streams sem --robust
echo "$(date) QUEUE2 DONE" >> results/queue.log
