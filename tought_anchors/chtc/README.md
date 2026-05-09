# Running Thought Anchors on CHTC

This repository now includes a minimal CHTC workflow for HTCondor jobs:

- `package_repo.sh` creates a single tarball of the repository, which matches CHTC's recommendation to compress many small files before transfer.
- `build_python_bundle.sh` installs Python packages into a relocatable directory bundle instead of a virtualenv, which is easier to move onto execute nodes.
- `run_repo_job.sh` unpacks everything into the job scratch directory and sets Hugging Face, datasets, sentence-transformers, and matplotlib caches to scratch.
- `submit_cpu.sub` and `submit_gpu.sub` are ready-to-edit submit file templates.
- `submit_cpu.sh` and `submit_gpu.sh` are thin convenience wrappers around `condor_submit`.

## 1. Build the Python dependency bundle

Run this on a CHTC submit node or inside an interactive CHTC build job so compiled wheels match the execute environment:

```bash
./chtc/build_python_bundle.sh
```

This creates `chtc/python-bundle.tar.gz`.

The build script keeps `pip`, `uv`, and temporary install files under `chtc/build/` in your home directory instead of inheriting a submit-node `TMPDIR` that may point at `/tmp`. This helps avoid `OSError: [Errno 28] No space left on device` during large installs.

For ad hoc installs outside these scripts, keep your shell temp directory in home too:

```bash
export TMPDIR="$HOME/.tmp"
```

If you want a GPU-enabled PyTorch build for `--provider Local`, set a PyTorch wheel index before running the script. `pip` honors environment variables like `PIP_EXTRA_INDEX_URL`.

## 2. Package the repository

```bash
./chtc/package_repo.sh
```

This creates `chtc/thought-anchors.tar.gz`.

## 3. Add API keys if needed

If your run needs API credentials, copy the example file and fill in only the keys you need:

```bash
cp chtc/job.env.example job.env
```

Then add `job.env` to `transfer_input_files` in your submit file. The wrapper automatically loads either `job.env` from the scratch directory or `.env` from the repo bundle.

## 4. Submit a job

CPU example:

```bash
condor_submit chtc/submit_cpu.sub
```

GPU example:

```bash
condor_submit chtc/submit_gpu.sub
```

The default GPU submit file is set up as a small smoke test for local inference:

- script: `generate_rollouts.py`
- provider: `Local`
- model: `Qwen/Qwen2.5-Math-7B`
- workload: `--num_problems 1 --num_rollouts 2`
- generation flags: `--batch_size 1 --max_tokens 4096`
- resources: `1 GPU`, `4 CPUs`, `32GB RAM`, `50GB disk`
- log layout: `chtc/logs/<cluster>.<process>/<cluster>.<process>.{log,out,err}`

## Prepared 6-job matrix

The repository also includes a prepared 6-job matrix for:

- `MATH` x `Qwen/Qwen2.5-Math-7B`
- `MATH` x `Qwen/Qwen2.5-Math-7B-Instruct`
- `MATH` x `Qwen/Qwen2.5-Math-7B-Oat-Zero`
- `MMLU` x `Qwen/Qwen2.5-Math-7B`
- `MMLU` x `Qwen/Qwen2.5-Math-7B-Instruct`
- `MMLU` x `Qwen/Qwen2.5-Math-7B-Oat-Zero`

Files involved:

- `job_matrix.json`: editable job matrix and resource defaults
- `render_submit_matrix.py`: renders concrete submit files under `chtc/generated/`
- `submit_prepared_jobs.sh`: helper that submits all 6 generated jobs

Render or re-render the prepared jobs with:

```bash
python3 chtc/render_submit_matrix.py
```

Submit all 6 prepared jobs with:

```bash
bash chtc/submit_prepared_jobs.sh
```

Before submitting, edit:

- `arguments` to the exact script and flags you want to run.
- `transfer_output_files` so it matches the directories your command creates under `repo/`, because the wrapper runs from the unpacked repository directory.
- `request_memory`, `request_disk`, `request_cpus`, and optionally `request_gpus` for your workload size.
- `transfer_input_files` if you need `job.env` or extra inputs.

## Notes

- The wrapper keeps caches in `_CONDOR_SCRATCH_DIR`, so model downloads and tokenizer caches do not spill into your home directory.
- The package and build helpers write their temporary files under `chtc/build/tmp` by default, not `/tmp`.
- `submit_gpu.sub` includes `+WantGPULab = true`, `+GPUJobLength`, and `request_gpus = 1`, following the current CHTC GPU guidance.
- If you are moving large datasets or model files, prefer CHTC staging storage instead of putting them into the repo tarball.

Relevant CHTC docs:

- File transfer: https://chtc.cs.wisc.edu/uw-research-computing/htc-job-file-transfer
- HTCondor submit files: https://chtc.cs.wisc.edu/uw-research-computing/htcondor-job-submission
- GPU jobs: https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs
