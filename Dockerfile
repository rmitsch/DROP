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
	
RUN	apt-get update && \
	apt-get -y install apt-utils && \
	apt-get -y install cmake && \
	# Allow execution of setup scripts.
	chmod +x /tmp/setup.sh && \
	# Install BLAS and LAPACK.
	apt-get -y install libblas-dev liblapack-dev && \
	# Install python dependencies.
	cat /tmp/requirements.txt | xargs pip install && \
	# Execute additional setup.
	./tmp/setup.sh

##########################################
# 3. Launch server.
##########################################

# Expose and launch only if this is supposed to run frontend.
# EXPOSE 2483
# CMD ["python", "./source/app.py"]