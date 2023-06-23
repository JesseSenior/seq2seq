#!/bin/bash

tmux new-session -d \
    'gpustat -i 1' \;\
    new-window 'python assess.py configs/example-hs128.conf > run/example-hs128/assess.log    ' \;\
    new-window 'python assess.py configs/example-hs256.conf > run/example-hs256/assess.log    ' \;\
    new-window 'python assess.py configs/example-hs256-ntf.conf > run/example-hs256-ntf/assess.log    ' \;\
    new-window 'python assess.py configs/example-hs512.conf > run/example-hs512/assess.log    ' \;\

tmux a -t 0:0