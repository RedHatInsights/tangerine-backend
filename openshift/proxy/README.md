# Proxy
An OAuth and TLS proxy for tangerine

# About
This provides 2 simple proxies that work together to allow Tangerine to run securely on OpenShift and be exposed to the internet. The first proxy is [OpenShift OAuth Proxy](https://github.com/openshift/oauth-proxy) which handles authentication and the OAuth flow. After authentication it routes requests to its upstream: the tangerine-proxy which is a Caddy reverse proxy running as a seperate container in the same pod. The tangerine-proxy routes requests to the frontend and backend pods. This ensures that both the frontend app and backend APIs are behind OAuth, without either app needing to implement the OAuth flow themselves.

Inspired by the [firelink-proxy](https://github.com/RedHatInsights/firelink-proxy)

# Deploying

```
NAMESPACE=<namespace>

oc process \
    -p SECRET=$(python -c 'import os,base64; print(base64.b64encode(os.urandom(16)).decode())') \
    -f secret.yaml | oc apply -f - -n $NAMESPACE

oc process -p HOSTNAME=<your public Route hostname> -f template.yaml | oc apply -f - -n $NAMESPACE
```
