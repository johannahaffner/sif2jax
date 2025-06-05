# Base
FROM ubuntu:latest AS base

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gfortran \
    git \
    python3-dev \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Build
# Create a simple script that outputs architecture information
FROM base AS build
WORKDIR /opt/cutest

# Clone the repositories and create a cache folder
RUN git clone --depth 1 https://github.com/ralna/ARCHDefs.git ./archdefs && \
    git clone --depth 1 https://github.com/ralna/SIFDecode.git ./sifdecode && \
    git clone --depth 1 https://github.com/ralna/CUTEst.git ./cutest && \
    git clone --depth 1 https://bitbucket.org/optrove/sif.git ./mastsif

# Set environment variables
ENV ARCHDEFS=/opt/cutest/archdefs
ENV SIFDECODE=/opt/cutest/sifdecode  
ENV CUTEST=/opt/cutest/cutest
ENV MASTSIF=/opt/cutest/mastsif
ENV MYARCH=pc64.lnx.gfo

# Run the PyCUTEst installation script
RUN curl -fsSL https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh > install_cutest.sh
RUN chmod +x install_cutest.sh
RUN /bin/bash -c ./install_cutest.sh

# Verify CUTEst installation
RUN cd $SIFDECODE/src ; make -f $SIFDECODE/makefiles/$MYARCH test
RUN cd $CUTEST/src ; make -f $CUTEST/makefiles/$MYARCH test

# Final stage
FROM base AS final
WORKDIR /opt/cutest
COPY --from=build /opt/cutest /opt/cutest

## Set the same environment variables as above - these are needed for pycutest to work correctly
ENV ARCHDEFS=/opt/cutest/archdefs
ENV SIFDECODE=/opt/cutest/sifdecode
ENV CUTEST=/opt/cutest/cutest
ENV MASTSIF=/opt/cutest/mastsif
ENV MYARCH=pc64.lnx.gfo

RUN mkdir -p pycutest_cache
ENV PYCUTEST_CACHE=/opt/cutest/pycutest_cache

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install jax everything with pip and uv
RUN pip install uv
RUN uv pip install jax jaxtyping equinox optimistix beartype hatchling pytest pycutest
RUN uv cache clean && pip cache purge

# Pre-cache the CUTEst problems to speed up our test runs
#COPY cache_pycutest.py .
#RUN python3 cache_pycutest.py

CMD ["/bin/bash"]
