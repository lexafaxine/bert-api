docker image build \
  -f docker/python3.6.dockerfile \
  -t ubuntu20.python3.6:latest .

docker image build \
  -f docker/mecab.dockerfile \
  -t "mecab-python3":latest .

docker image build \
  --no-cache \
  -f docker/server.dockerfile \
  -t "bert-server":latest .