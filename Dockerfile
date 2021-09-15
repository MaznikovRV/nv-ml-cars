FROM python:3.7-slim
ARG secret_key
ARG nv_storage_uri

COPY . /var/data/nv-ml-cars
RUN apt-get update && apt-get install -y git
RUN python3 -m venv venv
RUN /venv/bin/pip3 install --upgrade pip
RUN git config --global url.https://gitlab-ci-token:${secret_key}@gitlab.com/.insteadOf ssh://git@gitlab.com/
RUN /venv/bin/pip3 install -r /var/data/nv-ml-cars/requirements.txt
RUN /venv/bin/python3 /var/data/nv-ml-cars/cars_recognition.py check --nv_storage_uri ${nv_storage_uri};
WORKDIR /var/data/nv-ml-cars

ENTRYPOINT [ "/venv/bin/python3" ]


