FROM ubuntu:18.04
LABEL maintainer "SFL Scientific <dwang@sflscientific.com>"

# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # SSH and RDMA
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    iproute2 && \
    # Others
    apt-get install -y \
    build-essential \
    bzip2 \
    git \
    wget \
    cpio && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Conda Environment
ENV MINICONDA_VERSION 4.5.11
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

USER root:root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

COPY requirement.txt /requirement.txt
RUN pip install -r requirement.txt

RUN mkdir -p /SFL-Template
WORKDIR /SFL-Template

EXPOSE 8888

ENTRYPOINT ["bash", "./run.sh"]
CMD ["train"]