FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install software-properties-common git curl wget sudo -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install python3.6 python3-pip -y
RUN ln -s /usr/bin/python3.6 /usr/bin/python
RUN apt install mecab libmecab-dev mecab-ipadic-utf8 -y
RUN pip3 install mecab-python3
RUN git clone https://github.com/neologd/mecab-ipadic-neologd.git && cd mecab-ipadic-neologd && ./bin/install-mecab-ipadic-neologd -n -y
RUN mkdir -p /usr/local/etc/mecabrc
RUN pip3 install flask tensorflow==1.13.1 transformers==3.5.1
RUN git clone https://github.com/lexafaxine/bert-api.git