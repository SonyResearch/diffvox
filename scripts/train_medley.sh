#!/bin/bash

txt="${1%/}"
log="${2%/}"

cat $txt | while read song_id
do
    echo $song_id
    song_log="$log/$(basename $song_id)"
    mkdir -p $song_log
    python main.py data_dir=$song_id log_dir=$song_log dataset=medley_vocal || continue
done