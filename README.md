## 环境安装

``` bash
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

if render:

``` bash
apt-get install python-opengl
```

if mpe:

``` bash
pip install gym==0.25.1
```

## 工具

Xvfb :10 -screen 0 1024x768x16 &

export DISPLAY=:10

pkill -9 Xvfb

## 实验脚本

见/scripts文件夹下

| 方法 | 另名 | 脚本名 |
| ---- | ---- | ---- |
| TransfQmix-task | transfqmixtask | ExpRewardNormTransfqmixTask |
| TransfQmix-multihead | transfqmix_multihead | ExpRewardNormTransfqmixHead |
| TransfQmix-me | transfqmixme | ExpRewardNormTransfqmixMeTrue |
| MLPQmx-me | mixofencoders | ExpRewardNormTransfqmixMe |
| TETQmix (ours) | V1+imp | ExpRewardNormOursImp |
| TETQmix (w/o reg) | V1 | ExpRewardNormOursV1 |
| TETQmix (w/o first cab) | V1_without_first_cab+imp | ExpRewardNormOursV1ImpWithoutFirstCab |
| TETQmix (w/o second cab) | V1_without_second_cab+imp | ExpRewardNormOursV1ImpWithoutSecondCab |
| TETQmix (w/o sab) | V1_without_sab+imp | ExpRewardNormOursV1ImpWithoutSab |
