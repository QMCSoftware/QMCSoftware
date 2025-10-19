FROM condaforge/miniforge3:latest
SHELL ["/bin/bash", "-lc"]
WORKDIR /opt/QMCSoftware

# Copy the entire repository first so that '-e .' in environment.yml resolves correctly
COPY . /opt/QMCSoftware

RUN mamba install -c conda-forge gcc -y

# Use mamba if available, else conda
RUN if command -v mamba >/dev/null 2>&1; then \
      mamba env create -f environment.yml; \
    else \
      conda env create -f environment.yml; \
    fi && conda clean -afy

# Activate env by default
ENV PATH="/opt/conda/envs/qmcpy/bin:${PATH}"

# Smoke test
RUN python - <<'PY'
import qmcpy
print('qmcpy import OK:', getattr(qmcpy, '__version__', 'unknown'))
PY
