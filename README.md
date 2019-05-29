
# Environment setup
## docker

First, install [docker](https://docs.docker.com/install/#server) and [docker-compose](https://docs.docker.com/compose/install/).

__docker-compose must have minimum version 1.19.0 to use nvidia runtime__

Next, make sure the data you want to access from the container is in the directories specified in the `volumes:` section of `docker-compose.yml`. The data path you want to attach to the docker container __must__ be an absolute path (e.g. use `/home/wanglab/Desktop`, not `./Desktop`). Ideally, attach all local volumes to the `/opt/out` directory in the docker container.

To create the container:
```
sudo docker-compose build
```

__When there is a change to `docker-compose.yml` or `build/apnea.Dockerfile`, the container must be rebuilt.__

To start the built container and open into a bash prompt:
```
sudo docker-compose run mela-nn bash
```

-----------------
__ATTN:__ _Use simple\_\*.py unless you want a specific, known feature from the leagacy code_
