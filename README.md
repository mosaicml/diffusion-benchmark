# MosaicML Benchmark for Stable Diffusion

This repo provides code for benchmarking Stable Diffusion using [Streaming](https://github.com/mosaicml/streaming), [Composer](https://github.com/mosaicml/composer/tree/dev/composer), and [MosaicML Cloud](https://www.mosaicml.com/blog/mosaicml-cloud-demo). The benchmarking results are presented in [this blog post](https://www.mosaicml.com/blog/training-stable-diffusion-from-scratch-costs-160k), but the table is duplicated below.

| Number of A100s | Throughput (images / second)   | Days to Train on MosaicML Cloud | A100-hours | Approx. Cost on MosaicML Cloud |
| --------------- | ------------------------------ | ------------------------------- | ---------- | ------------------------------ |
| 8               | 128.2                          | 258.83                          | 49,696     | $99,000                        |
| 16              | 254.0                          | 130.63                          | 50,166     | $100,000                       |
| 32              | 485.7                          | 68.33                           | 52,470     | $105,000                       |
| 64              | 912.2                          | 36.38                           | 55,875     | $110,000                       |
| 128             | 1618.4                         | 20.5                            | 62,987     | $125,000                       |
| 256             | 2,589.4                        | 12.83                           | 78,735     | $160,000

In this repo, you will find:
- `benchmark.py` - defines the Stable Diffusion `ComposerModel` and the Composer `Trainer`.
- `data.py` - defines the MosaicML Stremaing LAION dataset and a synthetic dataset as an alternative to streaming data.
- `ema.py` - a memory-efficient version of Composer's [EMA algorithm](https://docs.mosaicml.com/en/v0.12.0/method_cards/ema.html)
- `mcloud.yaml` - examples of how to use MosaicML Cloud to launch a training run.

If you are interested in using the MosaicML Cloud, sign up for a demo [here](https://forms.mosaicml.com/demo)!

# Prerequistes

Install required dependencies using `pip install -r requirements.txt`

If you would like to use [xFormers](https://github.com/facebookresearch/xformers) install it using (we specify a commit we know will work):

```bash
pip install -v -U git+https://github.com/facebookresearch/xformers.git@3df785ce54114630155621e2be1c2fa5037efa27#egg=xformers
```

# Benchmarking

To benchmark without using a streaming dataset:

```bash
composer benchmark.py --use_ema --use_synth_data --device_train_microbatch_size 4
```

`device_train_microbatch_size` should be 4 when using a NVIDIA 40GB A100 GPUs and xFormers. If you are not using xFormers, `device_train_microbatch_size` should be 2. If using a smaller GPU, adjust `device_train_microbatch_size` as needed

To log benchmark results, set up a Weights and Biases account, then specify the `--wandb_name` and `--wandb_project` arguments.

If you want to benchmark using a streaming dataset, specify the `--remote` argument:

```bash
composer benchmark.py --use_ema --device_train_microbatch_size 4 --remote s3://my-bucket/laion/mds
```

# Contact Us

If you run into any problems with the code, please file Github issues directly to this repo.

If you want train diffusion models on MosaicML Cloud, [schedule a demo online](https://forms.mosaicml.com/demo) or email us at demo@mosaicml.com