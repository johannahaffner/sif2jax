# Base
FROM ubuntu:latest AS base

RUN apt-get update && apt-get install build-essential -y
RUN apt-get update && apt-get install curl -y
RUN apt-get update && apt-get install gfortran -y
RUN apt-get update && apt-get install git -y
RUN apt-get update && apt-get install python3-venv -y

# Build
# Create a simple script that outputs architecture information
FROM base AS build
WORKDIR /opt/cutest

# Clone the repositories and create a cache folder
RUN git clone https://github.com/ralna/ARCHDefs.git ./archdefs
RUN git clone https://github.com/ralna/SIFDecode.git ./sifdecode
RUN git clone https://github.com/ralna/CUTEst.git ./cutest
RUN git clone https://bitbucket.org/optrove/sif.git ./mastsif

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

# Now add python test-time dependencies on top
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir -p pycutest_cache
ENV PYCUTEST_CACHE=/opt/cutest/pycutest_cache
RUN pip install pycutest

RUN pip install pytest
RUN pip install jax
RUN pip install jaxtyping
RUN pip install equinox
RUN pip install optimistix
RUN pip install beartype

CMD ["/bin/bash"]
