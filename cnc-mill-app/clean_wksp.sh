docker container rm -f $(docker container ls -a -q)
docker image rm -f $(docker image ls -q)