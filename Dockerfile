FROM continuumio/miniconda3

# Note: Dockerfile has to be build from parent directory with
# docker build -t TALE -f DROP-backend/<dir/dir/Dockerfile> .


##########################################
# 1. Copy relevant files into container.
##########################################

# Copy file with python requirements into container.
COPY setup/environment.yml /tmp/environment.yml
# Copy setup file.
COPY setup/setup.sh /tmp/setup.sh
# Copy corrected base.py file.
COPY setup/base.py /tmp/base.py
# Copy source code.
COPY source /source
# Don't copy data, since we link it as a volume anyway.
# COPY data /data

# Set python path.
ENV PYTHONPATH /source

##########################################
# 2. Install dependencies.
##########################################

RUN conda env create --name envname --file=environments.yml

#RUN	apt-get update  && \
#	apt-get -y --no-install-recommends install apt-utils && \
#	apt-get -y --no-install-recommends install cmake  && \
#	apt-get -y --no-install-recommends install nano  && \
#   apt-get -y --no-install-recommends install libhdf5-serial-dev && \
#	# Allow execution of setup scripts.
#	chmod +x /tmp/setup.sh  && \
#	# Install BLAS and LAPACK.
#	apt-get -y --no-install-recommends install libblas-dev liblapack-dev  && \
#	apt-get -y --no-install-recommends install nano && \
#	# Install python dependencies.
#	cat /tmp/requirements.txt | xargs pip install  && \
#	# Execute additional setup.
#	./tmp/setup.sh  && \
#	#
#	# Clean-up.
#	#
#	apt-get remove --purge -y cmake  && \
#	apt-get remove --purge -y apt-utils && \
# 	apt-get autoremove -y && \
#	rm -rf /var/lib/apt/lists/*

##########################################
# 3. Launch server.
##########################################

# Expose and launch only if this is supposed to run frontend.
EXPOSE 2483
CMD ["python", "./source/app.py"]
