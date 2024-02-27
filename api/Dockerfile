FROM node:14.15.1-alpine3.11 as builder
WORKDIR /usr/src/app
COPY ./ ./

RUN npm i \
    && npm run clean \
    && npm run build-ts \
    && rm -rf node_modules

FROM builder
ENV TZ=Europe/Moscow
RUN addgroup -S app && adduser -S app -G app
ENV HOME=/home/app
ENV APP_HOME=/home/app/api

WORKDIR $APP_HOME
RUN apk --no-cache add curl
COPY --from=builder /usr/src/app/ ./
RUN npm i
