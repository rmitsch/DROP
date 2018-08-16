# Note: Container size can be reduced later.
FROM python:3.5

##########################################
# 1. Copy relevant files into container.
##########################################

# Copy file with python requirements into container.
COPY setup/requirements.txt /tmp/requirements.txt
# Copy setup file.
COPY setup/setup.sh /tmp/setup.sh
# Copy source code.
COPY source /source
# Copy data.
COPY data /data

# Set python path.
ENV PYTHONPATH /source

##########################################
# 2. Install dependencies.
##########################################
	
RUN	apt-get update  && \
	apt-get -y --no-install-recommends install apt-utils && \
	apt-get -y --no-install-recommends install cmake  && \
	apt-get -y --no-install-recommends install nano  && \
	# Allow execution of setup scripts.
	chmod +x /tmp/setup.sh  && \
	# Install BLAS and LAPACK.
	apt-get -y --no-install-recommends install libblas-dev liblapack-dev  && \
	apt-get -y --no-install-recommends install nano && \
	# Install python dependencies.
	cat /tmp/requirements.txt | xargs pip install  && \
	# Execute additional setup.
	./tmp/setup.sh  && \
	#
	# Clean-up.
	#
	apt-get remove --purge -y cmake  && \
	apt-get remove --purge -y apt-utils && \
 	apt-get autoremove -y && \
	rm -rf /var/lib/apt/lists/*

##########################################
# 3. Launch server.
##########################################

# Expose and launch only if this is supposed to run frontend.
EXPOSE 2483
CMD ["python", "./source/app.py"]
