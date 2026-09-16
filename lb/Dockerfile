FROM nginx:1.27-alpine

RUN apk add --no-cache openssl curl

COPY nginx.conf /etc/nginx/nginx.conf.template
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80 443
ENTRYPOINT ["/entrypoint.sh"]
