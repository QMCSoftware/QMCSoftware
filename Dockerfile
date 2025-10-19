# syntax=docker/dockerfile:1.7
ARG TARGETPLATFORM
ARG BUILDPLATFORM
FROM --platform=$TARGETPLATFORM mambaorg/micromamba:1.5.10

# Activate this env at runtime
ENV ENV_NAME=qmcpy
WORKDIR /opt/QMCSoftware

# Copy environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Install Python + PyYAML in base to transform YAML
RUN micromamba install -y -n base -c conda-forge python=3.11 pyyaml && \
    micromamba clean -a -y

# Split conda deps from pip deps; drop editable '-e .'
RUN python - <<'PY'
import yaml, pathlib
src = pathlib.Path("/tmp/environment.yml")
env = yaml.safe_load(src.read_text())
deps = env.get("dependencies", [])
conda_deps, pip_pkgs = [], []
for d in deps:
    if isinstance(d, str):
        conda_deps.append(d)
    elif isinstance(d, dict) and "pip" in d:
        for p in (d["pip"] or []):
            if isinstance(p, str) and p.strip() != "-e .":
                pip_pkgs.append(p)
env["dependencies"] = [d for d in conda_deps if not (isinstance(d, str) and d.strip() == "-e .")]
pathlib.Path("/tmp/environment.conda.yml").write_text(yaml.safe_dump(env, sort_keys=False))
pathlib.Path("/tmp/requirements-pip.txt").write_text("\n".join(pip_pkgs) + ("\n" if pip_pkgs else ""))
PY

# Create the conda env
RUN micromamba env create -f /tmp/environment.conda.yml && micromamba clean -a -y

# Ensure RUN uses the newly created env
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# On amd64, preinstall CPU PyTorch so gpytorch/botorch resolve cleanly
RUN if [[ "$TARGETPLATFORM" == "linux/amd64" ]]; then \
      python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu "torch>=1.13,<2.1"; \
    fi

# Copy the repo after env solve to preserve layer cache
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Install pip-only deps and then this package
RUN if [[ -s /tmp/requirements-pip.txt ]]; then \
      python -m pip install --no-cache-dir -r /tmp/requirements-pip.txt; \
    fi && \
    python -m pip install --no-cache-dir -e .

# Smoke test
RUN python - <<'PY'
import sys
print("Python:", sys.version)
try:
    import qmcpy
    print("qmcpy import ok")
except Exception as e:
    print("qmcpy import failed:", e)
    raise
PY

CMD ["python","-c","import qmcpy,sys; print('qmcpy ready on', sys.platform)"]
