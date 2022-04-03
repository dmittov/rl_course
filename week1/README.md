# Week 1: Introduction

[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/zE8zNCzJ)
[![Checked with black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://camo.githubusercontent.com/59eab954a267c6e9ff1d80e8055de43a0ad771f5e1f3779aef99d111f20bee40/687474703a2f2f7777772e6d7970792d6c616e672e6f72672f7374617469632f6d7970795f62616467652e737667)](http://mypy-lang.org/)

## Environment

* ```bash
  conda create -n week1 -f environment.yaml
  conda activate week1
  ```

## Usage

* Train

```bash
python train.py
```

* Play with trained agent

```bash
python play.py
```

Trained agent is serialized by hardcoded path: agent.data
