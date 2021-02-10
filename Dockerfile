FROM ubuntu:18.04

LABEL Oskari Honkasalo "oskari.honkasalo@axwdigital.fi"


RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# Get pip3
RUN apt-get -y install python3-pip


# XGBoost dependency
RUN apt-get update && \
     apt-get -y --no-install-recommends install \
     libgomp1


# Install GDAL and GEOS and CBLAS
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntugis/ppa && apt-get update
RUN apt-get update
RUN apt-get install -y gdal-bin 
RUN apt-get install -y libgdal-dev
ARG CPLUS_INCLUDE_PATH=/usr/include/gdal
ARG C_INCLUDE_PATH=/usr/include/gdal
RUN apt-get install libgeos-dev
RUN apt install -y libcurl4-gnutls-dev librtmp-dev
RUN pip install Cython
RUN apt-get install -y libclblas2


# We copy just the requirements.txt first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
WORKDIR /app


RUN pip install -r requirements.txt
RUN pip3 install sqlalchemy-datatables==2.0.1

COPY . /app

ADD . /

ENTRYPOINT [ "python" ]

CMD [ "launch.py", "--host", "0.0.0.0"]

EXPOSE 5000