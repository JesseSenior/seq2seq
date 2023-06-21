#!/bin/bash

tmux new-session -d \
    'gpustat -i 1' \;\
    new-window 'python -u train.py ./configs/example-hs128.conf    ' \;\
    pipe-pane -o 'cat >> log-example-hs128.log' \;\
    new-window 'python -u train.py ./configs/example-hs256.conf    ' \;\
    pipe-pane -o 'cat >> log-example-hs256.log' \;\
    new-window 'python -u train.py ./configs/example-hs256-ntf.conf' \;\
    pipe-pane -o 'cat >> log-example-hs256-ntf.log' \;\
    new-window 'python -u train.py ./configs/example-hs512.conf    ' \;\
    pipe-pane -o 'cat >> log-example-hs512.log' 

tmux a -t 0:0
 