#!/bin/bash
# Launch an experiment using the docker cpu image

docker run --ipc=host --rm -it \
  -v /home/zheong/slimevolleygym/:/volleyball/ \
  stablebaselines/rl-baselines-zoo-cpu:v2.10.0 \
  /bin/bash

