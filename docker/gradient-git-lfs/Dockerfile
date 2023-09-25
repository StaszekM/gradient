FROM node:18-alpine

RUN npm install -g node-git-lfs

WORKDIR /gradient-git-lfs

COPY ./config.json /gradient-git-lfs/config/default.json

ENTRYPOINT [ "/bin/sh", "-c", "npx node-git-lfs" ]
