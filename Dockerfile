FROM nvidia/cuda:9.0-devel as cuda9.0-runtime

FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY requirements.txt requirements.txt

COPY --from=cuda9.0-runtime /usr/local/cuda-9.0 /usr/local/cuda-9.0

RUN apt-get update 

RUN pip install --upgrade pip
RUN pip install git+https://github.com/wookayin/gpustat.git@master
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.1/lib64:/usr/local/cuda-9.0/lib64

EXPOSE 5000