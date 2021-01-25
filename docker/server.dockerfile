FROM mecab-python3:latest
RUN pip3 install flask tensorflow==1.13.1 transformers==3.5.1
RUN git clone --recursive https://github.com/lexafaxine/bert-api.git
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV FLASK_APP main.py
ENV FLASK_ENV development
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT 5000
ENV MECAB_DICT_PATH "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"

WORKDIR /bert-api
CMD flask run