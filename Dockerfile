FROM docker.io/nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Base system packages
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        python3 python3-pip build-essential git \
        gdal-bin libgdal-dev proj-bin libproj-dev \
        sudo && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python dependency manager (uv)
# -----------------------------------------------------------------------------
RUN pip install --upgrade pip && pip install uv

# -----------------------------------------------------------------------------
# Create unprivileged user (UID/GID will be rewritten by dev‑containers)
# -----------------------------------------------------------------------------
ARG USERNAME=vscode
RUN useradd -m "$USERNAME" && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# -----------------------------------------------------------------------------
# Workspace setup
# -----------------------------------------------------------------------------
WORKDIR /workspace
USER ${USERNAME}

CMD ["/bin/bash"]

ENV PYTHONPATH=/workspaces/mtbs_fire_analysis