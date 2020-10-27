FROM python:3.8
WORKDIR /code
RUN pip install numpy==1.18.5
RUN pip install scipy==1.4.1
COPY qmcpy/ ./qmcpy/
COPY setup.py .
RUN pip install -e .
