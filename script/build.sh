export PROJECT_BASE=$(dirname $(dirname $(readlink -f "$0")))

docker image build \
  -f ${PROJECT_BASE}/docker/python3.6.dockerfile \
  -t ubuntu20.python3.6:latest .

docker image build \
  -f ${PROJECT_BASE}/docker/mecab.dockerfile \
  -t "mecab-python3":latest .

docker image build \
  --no-cache \
  -f ${PROJECT_BASE}/docker/server.dockerfile \
  -t "bert-server":latest .