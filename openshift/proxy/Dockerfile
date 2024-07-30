  FROM quay.io/cloudservices/caddy-ubi:latest

  ENV CADDY_TLS_MODE http_port 8000

  COPY ./Caddyfile /opt/app-root/src/Caddyfile

  CMD ["caddy", "run", "--config", "/opt/app-root/src/Caddyfile"]
