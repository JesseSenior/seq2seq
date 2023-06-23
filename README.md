# Seq2seq 中译英实现

实验要求：https://www.zybuluo.com/hxLau/note/2525331

源码部分参考自：https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#visualizing-attention

实验环境：

1. docker容器：https://github.com/JesseSenior/docker-ubuntu-enhanced
```shell
docker run \
    -d \
    --name 'DLCV' \
    -e AUTHORIZED_KEY='your SSH pub key' \
    -p 3333:22 \
    --gpus all \
    --shm-size 4G \
    --restart unless-stopped \
    --device /dev/nvidia0:/dev/nvidia0 \
    --device /dev/nvidia1:/dev/nvidia1 \
    --device /dev/nvidia2:/dev/nvidia2 \
    --device /dev/nvidia-modeset:/dev/nvidia-modeset \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
    --device /dev/nvidiactl:/dev/nvinvidiactl  \
    ubuntu-enhanced:latest
```
2. 显卡
    1. GPU 0: Tesla P100-PCIE-12GB
    2. GPU 1: Tesla P100-PCIE-16GB
    3. GPU 2: Tesla P100-PCIE-16GB
