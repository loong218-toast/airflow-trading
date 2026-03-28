# Use the version currently in your yaml
FROM apache/airflow:3.1.7

USER root
# This is the specific fix for the 'libgomp.so.1' error
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt