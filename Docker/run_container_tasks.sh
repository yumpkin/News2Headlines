docker-compose run --user $(id -u) --rm mld07_transformers \
--task="sentiment" \
--document="I don't hate my dog - just the opposite, I love it very much!"

docker-compose run --user $(id -u) --rm mld07_transformers \
--task="summariztion" \
--document="John was sitting on a porch with his tea and a newspaper. He was thinking about what happened yesterday trying to understand it. Suddenly"
