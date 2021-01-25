export TF_SERVER_HOST="tf-serving:8501"

docker network create -d bridge bert-net

docker run --detach --rm \
  --mount type=bind,source=$(dirname $(dirname $(readlink -f "$0")))/saved_model,target=/models/ner_32k \
  -e MODEL_NAME=ner_32k \
  --name tf-serving \
  --network bert-net \
  -t tensorflow/serving

docker run --detach --rm \
  -p 12345:5000 \
  --network bert-net \
  --name bert-api \
  --env TF_SERVER_HOST=${TF_SERVER_HOST} \
  bert-server
