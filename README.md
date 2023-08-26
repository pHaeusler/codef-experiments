# CoDeF - experiments

A smaller, simpler, and faster version of <https://github.com/qiuyu96/CoDeF>

I built this to understand capabilities and limitations of the approach

- batched training (much faster ~1min train)
- only 1x canonical/warping model (no background models)
- no masks
- optical flow computed with `cv::calcOpticalFlowFarneback` (rather than [RAFT](https://github.com/princeton-vl/RAFT))
- no config files

**Train**

```bash
python3 run.py train --image_dir ./beauty_1
```

**Train with a frame as the canonical image**

```bash
python3 run.py train --image_dir ./beauty_1 --canonical ./beauty_1/00001.png
```

**Generate frames**

```bash
python3 run.py generate --checkpoint ./checkpoints/step=200.pt
```

**Generate frames with a new canonical image**

```bash
python3 run.py generate --checkpoint ./checkpoints/step=200.pt --canonical canonical.png
```
