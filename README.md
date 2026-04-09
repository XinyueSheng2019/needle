# NEEDLE2.0

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)](#)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-2026--04--08-lightgrey.svg)](#)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19484687.svg)](https://doi.org/10.5281/zenodo.19484687)

NEEDLE2.0 is a Python workflow for building datasets and training transient-object classifiers with the `needle_train` pipeline.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

1. Configure paths in `config.py`:
   - `DEFAULT_DATA_PATH`
   - `MAG_OUTPUT_PATH`
   - `HOST_DATA_PATH`
   - `OBJ_INFO_PATH`

2. Build train/validation datasets:

```bash
python needle_train/get_train_valid_sets.py
```

3. Train and evaluate model:

```bash
python needle_train/run_model.py
```

## Project Structure

- `needle_train/`: training, preprocessing, and dataset build scripts
- `image/`, `light_curve/`: image and photometry-related processing modules
- `config.py`: central configuration for paths and training options
- `utils.py`: shared helper functions

## Inputs and Outputs

- Inputs: external data roots configured in `config.py`, with optional reference tables under `info/`.
- Outputs:
  - `needle_inputs/` (`NEEDLE_SET_PATH`)
  - `light_curve/photo_processing_output_new/` (`PHOTO_OUTPUT_PATH`)
  - `image/image_preprocessing_output/` (`IMG_OUTPUT_PATH`)
  - `image/image_unmasked_output/` (`UNMASKED_IMG_OUTPUT_PATH`)
- Large generated outputs are ignored by `.gitignore` to keep commits small.

## License

This project is licensed under the MIT License. See `LICENSE` for details.


