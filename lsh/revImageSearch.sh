#!/bin/bash
id=$(sha256sum $1 | cut -d' ' -f1)
convert -resize 32x32! $1 $1
echo $id
mkdir -p images/$id 2>/dev/null
matches=$(python knn_lsh.py --db ../tiny_images.bin $1  | head -n10 | cut -d' ' -f1 | tr '\n' ' ')
echo $matches
python extract-images.py --db ../tiny_images.bin $matches --path images/$id 
