#!/bin/bash

docker run --name elasticsearch --net elastic -p 9200:9200 -e "discovery.type=single-node" -t elasticsearch:7.9.2