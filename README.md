# KITScenes API

Python API for the [KITScenes Multimodal](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) autonomous driving dataset.

**A high-fidelity sensor suite and the most complete HD maps of any public autonomous driving dataset.**

**Links:** [Dataset website](https://kitscenes.com/multimodal/) · [Download on HuggingFace](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) · [Python API on GitHub](https://github.com/KIT-MRT/kitscenes)

> **Early release.** KITScenes Multimodal is published on HuggingFace at version **1.0.x**. The on-disk schema is in place, but files, annotations, splits, and documentation may still change. For final benchmark reporting, please wait for a **more stable** public release.

<video src="https://huggingface.co/datasets/immel-f/KITScenes-Multimodal-Sample-Video/resolve/main/teaser_combined_for_huggingface.mp4" controls width="100%"></video>

*Reprojection of HD map labels into 6 of the 9 cameras, with LiDAR points reprojected into the rear cameras.*

KITScenes Multimodal is a European urban autonomous driving dataset targeting Level 4 robotaxi requirements, recorded across Karlsruhe, Frankfurt, and Sindelfingen by the Institute of Measurement and Control Systems (MRT) at Karlsruhe Institute of Technology (KIT). The dataset is built around a **high-fidelity sensor suite**: nine high-resolution global-shutter cameras provide full 360° surround coverage at 72.5 MPix per frame, enabling novel view synthesis and holistic HD map perception, while seven highly dense long-range LiDARs with an effective range beyond 400 m push the limits of what current perception methods can achieve.

KITScenes Multimodal provides what we believe to be **the most complete HD maps of any public autonomous driving dataset**, annotated in Lanelet2 format across 62 km² with full topological connectivity between lanes, signs, and traffic lights. The maps have been validated in closed-loop autonomous driving trials using the open-source Autoware stack — meaning they are ready to drive on directly, while simultaneously enabling research to close the gap between the current state of the art and the actual requirements of L4 robotaxi deployment.

| | | | |
|---|---|---|---|
| **1,000+** driving scenarios | **72.5 MPix** per frame · global shutter | **400+ m** effective LiDAR range | **62 km²** Lanelet2 HD maps |

This repository provides the Python `kitscenes` package for loading scenes, accessing synchronized sensor data, querying HD maps, and running visualization / download tooling against the dataset hosted on [HuggingFace](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal).

**Highlights**

- **European urban focus** — recordings from Karlsruhe, Frankfurt, and Sindelfingen
- **High-fidelity sensor suite** — 72.5 MPix global-shutter surround cameras, 7 LiDARs (~906k points/frame on average), 3 Continental ARS548 4D imaging radars, and redundant GNSS/INS
- **Long-range sensing** — effective LiDAR range beyond 400 m with substantially higher return density than common public driving datasets
- **Production-grade HD maps** — Lanelet2 maps with lane topology, regulatory elements, 29 road-feature classes, 120 traffic-sign classes (GTSIGN-220 taxonomy), and 3D-localized signs, traffic lights, and poles
- **Research benchmarks** — relational HD map perception, long-range monocular depth estimation, novel view synthesis, and end-to-end / world-model research

**License:** This repository (the `kitscenes` Python package) is licensed under [Apache-2.0](LICENSE). The [KITScenes Multimodal dataset](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) is separate — see [License](#license) below.

**Citation:** If you use KITScenes Multimodal in research, please cite [The Road Ahead in Autonomous Driving: The KITScenes Multimodal Dataset](https://arxiv.org/abs/2606.02956) (arXiv:2606.02956). BibTeX below.

## Versioning

### Dataset (HuggingFace)

The [HuggingFace dataset](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) follows [Semantic Versioning](https://semver.org) (`MAJOR.MINOR.PATCH`) for **data and on-disk layout**:

| Bump | Version | Meaning |
|------|---------|---------|
| **Major** | `X.0.0` | New dataset schema or file format — breaks backwards compatibility |
| **Minor** | `1.X.0` | New driving data — additional scenarios or recordings |
| **Patch** | `1.0.X` | Updated maps, labels, metadata, or calibrations within the existing schema — fully compatible, no new scenarios |

Releases are tagged on HuggingFace (e.g. `v1.0.1`) for reproducible experiments. Pin a tag or commit when downloading for benchmark work (current 1.0.1 release not recommended for benchmark reporting). See the [dataset README](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal#versioning) for release notes.

### Python API (this repository)

This package follows [Semantic Versioning](https://semver.org) for **code**. The current line is **`0.1.x`** (alpha pre-release): the public API is under active development and **may change without notice**. After `1.0.0`, semver applies to the API — breaking changes bump major, backward-compatible features bump minor, bug fixes bump patch.

<details>
<summary><strong>Release notes — Python API</strong></summary>

#### 0.1.0
- Alpha pre-release of the `kitscenes` package
- Dataset loading, sensor access, HD map queries, visualization, and download tooling

</details>

---

## Setup 

**Constraints:** Linux x86_64, Python 3.8–3.12, `numpy<2.0`. A pre-build beta version of Lanelet2 is provided under `res/ml_converter_wheels/` (not PyPI) and installed via the `[map]` or `[all]` extra. Map API needs it; everything else works without.


```bash
git clone https://github.com/KIT-MRT/kitscenes.git && cd kitscenes
pip install --upgrade pip
pip install -e ".[all]"
pytest test/ # test might fail depending on your installed extras
```

for a minimal install, this includes the custom lanelet2 wheels:
```bash
pip install -e ".[map]"
```

Dependencies and wheel paths are defined in `pyproject.toml`. List vendored wheels with `ls res/ml_converter_wheels/`.

---

## Download

1. **Accept the dataset terms on HuggingFace (gated): [KITScenes-Multimodal](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal)**
2. **Authenticate**: `hf auth login` (`huggingface-cli login` is deprecated on newer HF CLI)

```bash
export KITSCENES_ROOT=<YOUR_STORAGE_LOCATION>   # download location, inplace extraction or mounting later on.
```

Scenes land under `$KITSCENES_ROOT/data/<split>/<scene_id>/`. Total size is printed before extract; Ctrl+C to abort.

### Default — mirror + extract (recommendation for large machine local disk)
> recommended for a full dataset download
Fetch the repo with HuggingFace tooling, then verify and extract in parallel. Tars are deleted after each scene, so disk use stays near the **final extracted size plus a small buffer** while untar runs — not ~2× the dataset.

```bash
hf download KIT-MRT/KITScenes-Multimodal --repo-type dataset --local-dir "$KITSCENES_ROOT"
python -m kitscenes.download "$KITSCENES_ROOT" --extract-local --jobs 16
```

**Example Partial mirror (~5 GB)** — fetch a subset with `hf download`, then extract only what arrived:

> c34c778f-ad8c-0aa9-7e1a-c86a73f887c7 is the default example scene UUID used in the notebooks
```bash
hf download KIT-MRT/KITScenes-Multimodal \
    --repo-type dataset \
    --local-dir "$KITSCENES_ROOT" \
    --include "data/sequence_archives.csv" \
    --include "data/val/c34c778f-ad8c-0aa9-7e1a-c86a73f887c7.tar" 
    --max-workers 32

python -m kitscenes.download "$KITSCENES_ROOT" --extract-local --split val --jobs 32
```

Adjust the `--include` glob to control how much you pull.

### Alternative — built-in downloader (few samples / tight disk)
> not! recommended for a full dataset download

Downloads one tar at a time, extracts, deletes the tar. Slower, but disk grows incrementally — useful for a quick test or when you cannot mirror all archives at once. Use `--split` or `--max-gb` to cap what you fetch.

```bash
# sample scene (~5 GB)
python -m kitscenes.download "$KITSCENES_ROOT" \
    --scenes c34c778f-ad8c-0aa9-7e1a-c86a73f887c7 

# e.g. first 50 GB of val
python -m kitscenes.download "$KITSCENES_ROOT" --split val --max-gb 50
```

More flags: `python -m kitscenes.download --help`

### Direct mounting of .tar scene files with ratarmount 
> readonly. simple solution for network-storage / HPC-storage with limited inodes and network congesting filesystem metadata when using many small files

see: https://github.com/mxmlnkn/ratarmount#benchmarks

Keep mirrored `.tar` files on disk and skip full extract — useful after `hf download` when you retain archives (do not run `--extract-local`) and want to get started quickly.


```bash
hf download KIT-MRT/KITScenes-Multimodal --repo-type dataset --local-dir "$KITSCENES_ROOT" # download dataset first
pip install -e ".[mount]" # installs ratarmount
source scripts/mount_kitscenes_tars.sh "$KITSCENES_ROOT" # mounts tars for you and 
# exported KITSCENES_ROOT=/tmp/$USER/ratarmount/kitscenes
python -c "from kitscenes import KITScenesDataset; print(KITScenesDataset(split='val'))" # test it
python -c "from kitscenes import KITScenesDataset; print(KITScenesDataset(split='train'))" # test it
python -c "from kitscenes import KITScenesDataset; print(KITScenesDataset())" # test it
# Unmount when done:
# ratarmount -u /tmp/$USER/ratarmount/kitscenes/data/*
```

Each split is union-mounted from `data/<split>/*.tar` to `/tmp/$USER/ratarmount/kitscenes/data/<split>/`. Scene folders appear as `<scene_uuid>/` (tar basename without `.tar`). First mount per split builds ratarmount index files (beside the tars or in `~/.cache/ratarmount`) and can take a while on large splits. Unlike the SquashFS workflow below, no `pack_sqfs` / `mksquashfs` step is needed, but all per-scene tars remain on disk.

### SquashFS image 

> readonly. more complex than tar mounts but potentially faster, no full extract but inplace sqashfs creation

Build one read-only image **per split** with [ratarmount](https://github.com/mxmlnkn/ratarmount) + `mksquashfs` (tars removed after append). Each image contains `<scene_id>/` at its root. Uses **lz4** by default (`--comp none` to disable).

```bash
python -m kitscenes.pack_sqfs --processors 16
# -> $(dirname $KITSCENES_ROOT)/kitscenes_sqfs/train.sqfs, val.sqfs, ...
```

Mount to an ephemeral FUSE tree and **source** the script (sets `$KITSCENES_ROOT`):

```bash
source scripts/mount_kitscenes_sqfs.sh $KITSCENES_ROOT/../kitscenes_sqfs
# this exported KITSCENES_ROOT=/tmp/$USER/sqfs_mnt/kitscenes
python -c "from kitscenes import KITScenesDataset; print(KITScenesDataset(split='val'))"

# Unmount all sqfs mounts when done like this :
# fusermount -u /tmp/$USER/sqfs_mnt/kitscenes/data/train
```

Requires `ratarmount` + `mksquashfs` to build; `squashfuse` to mount (see `scripts/mount_kitscenes_sqfs.sh`). More flags: `python -m kitscenes.pack_sqfs --help`

---

## Quickstart

```python
from kitscenes import KITScenesDataset

ds = KITScenesDataset(split="train")  # uses $KITSCENES_ROOT
print(ds)
# iterate on a scene level
scene = ds[0]
print(scene)
```




<details>
<summary><strong>HD map Vizualization CLI (ML converter — traffic elements, styled lane markings):</strong></summary>
example CLI call (read-only inputs from dataset root; writes go to external output):

```bash
SCENE_UUID=c34c778f-ad8c-0aa9-7e1a-c86a73f887c7
SPLIT=val
export KITSCENES_VIZ_OUTPUT="./outputs/visualization/$SPLIT/$SCENE_UUID"
SCENE="$KITSCENES_ROOT/data/$SPLIT/$SCENE_UUID"

python -m kitscenes.visualization project \
  --base-dir "$SCENE" \
  --map-path "$SCENE/maps/map.osm"

python -m kitscenes.visualization video \
  --visdir "$KITSCENES_VIZ_OUTPUT"
```

**CLI help**:

```bash
python -m kitscenes.visualization --help          # list subcommands: project, video
python -m kitscenes.visualization project --help  # map-on-camera + top-down options
python -m kitscenes.visualization video --help    # MP4 assembly options
```

| Command | Notable options (see `--help` for full list) |
|---------|-----------------------------------------------|
| `project` | `--output-dir` or `$KITSCENES_VIZ_OUTPUT` (required; must not be under `$KITSCENES_ROOT`); UTM origin from `maps/origin.json` unless `--lat-origin` / `--lon-origin`; `--frame-step` (default 5); `--num-processes` (default 16); `--skip-top-down`, `--front-only`, `--grid-only`, `--no-generate-grid` |
| `video` | `--visdir` (grid + camera output from `project`); `--output-path` or `$KITSCENES_VIZ_OUTPUT` / `map_projection.mp4`; `--fps` (default 2) |

</details>

<details>
<summary><strong>Splits</strong></summary>

`$KITSCENES_ROOT` is always the dataset root. Pass `split=` to pick one split; omit it to load all scenes you have on disk (across splits).

```python
ds_train = KITScenesDataset(split="train")
ds_val = KITScenesDataset(split="val")
ds_all = KITScenesDataset()              # all downloaded scenes under $KITSCENES_ROOT
```

</details>

---

## Explore further

The `notebooks/` directory contains executed examples (set `DATASET_ROOT` or `$KITSCENES_ROOT` in each notebook):


| Notebook                              | Topic                                                                 |
| ------------------------------------- | --------------------------------------------------------------------- |
| `01_quickstart.ipynb`                 | Dataset loading, scenes, timestamps, ego poses, `FrameDataset`        |
| `02_sensor_data.ipynb`                | `SensorDataLoader`, LiDAR/radar deskewing, sensor visualizations      |
| `03_map_labels.ipynb`                 | ML-converter HD map labels, top-down BEV, map-on-camera projection    |
| `04_calibration_and_multimodal.ipynb` | Camera/LiDAR calibration, multi-LiDAR fusion, LiDAR/radar overlays    |


## License

### Software (this repository)

The `kitscenes` source code is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).

### Dataset (HuggingFace)

The KITScenes Multimodal **dataset** is **not** covered by Apache-2.0. Access is gated on HuggingFace. By requesting access you agree to use the data under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) together with the additional KITScenes terms on the [dataset page](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal). Where those dataset terms conflict with CC BY-NC 4.0, the dataset terms prevail.

Using this Python API does **not** grant rights to use the dataset beyond those terms.

---

## Citation

If you use KITScenes Multimodal in research, please cite:

**The Road Ahead in Autonomous Driving: The KITScenes Multimodal Dataset**  
[arXiv:2606.02956](https://arxiv.org/abs/2606.02956)

```bibtex
@misc{schwarzkopf2026kitscenes,
      title={The Road Ahead in Autonomous Driving: The KITScenes Multimodal Dataset}, 
      author={Richard Schwarzkopf and Fabian Immel and Alexander Blumberg and Jonas Merkert and Nils Rack and Kaiwen Wang and Fabian Konstantinidis and Julian Truetsch and Carlos Fernandez and Annika Bätz and Kevin Rösch and Marlon Steiner and Willi Poh and Yinzhe Shen and Royden Wagner and Felix Hauser and Dominik Strutz and Jaime Villa and Gleb Stepanov and Holger Caesar and Ömer Şahin Taş and Frank Bieder and Jan-Hendrik Pauls and Christoph Stiller},
      year={2026},
      eprint={2606.02956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.02956}, 
}
```

