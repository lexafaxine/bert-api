FROM ubuntu20.python3.6:latest
RUN apt-get install mecab libmecab-dev mecab-ipadic-utf8 -y
RUN pip3 install mecab-python3
RUN git clone https://github.com/neologd/mecab-ipadic-neologd.git && ./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y && rm -rf mecab-ipadic-neologd
RUN mkdir -p /usr/local/etc/mecabrc