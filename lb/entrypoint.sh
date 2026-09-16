#!/bin/sh
set -eu

CERT_DIR="${CERT_DIR:-/etc/nginx/certs}"
TLS_DAYS="${TLS_DAYS:-825}"
WEBUI_URL="${WEBUI_URL:-https://localhost}"

resolve_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s\n' "${CERT_DIR}/$1" ;;
    esac
}

CERT="$(resolve_path "${TLS_CERT:-fullchain.pem}")"
KEY="$(resolve_path "${TLS_KEY:-privkey.pem}")"
CANON_CERT="${CERT_DIR}/fullchain.pem"
CANON_KEY="${CERT_DIR}/privkey.pem"

mkdir -p "$CERT_DIR"

# Split WEBUI_URL into host + port. Container nginx always listens on 443;
# a non-443 port only affects the redirect Location and the cert marker.
parse_webui_url() {
    raw="$(printf '%s' "$1" | tr -d '[:space:]')"
    if [ -z "$raw" ]; then
        echo "lb: WEBUI_URL is empty" >&2
        exit 1
    fi

    case "$raw" in
        https://*) raw="${raw#https://}" ;;
        http://*)
            echo "lb: WEBUI_URL should be https://...; ignoring http://" >&2
            raw="${raw#http://}"
            ;;
        *://*)
            echo "lb: WEBUI_URL must be https://host[:port] (got $1)" >&2
            exit 1
            ;;
    esac

    raw="${raw%%/*}"
    raw="${raw%%\?*}"
    raw="${raw%%#*}"
    raw="${raw##*@}"

    HOST=""
    PORT="443"
    case "$raw" in
        \[*\]:*)
            HOST="${raw#\[}"
            HOST="${HOST%%\]*}"
            PORT="${raw##*\]:}"
            ;;
        \[*\])
            HOST="${raw#\[}"
            HOST="${HOST%\]}"
            ;;
        *:*)
            stripped="$(printf '%s' "$raw" | tr -d ':')"
            colons=$(( ${#raw} - ${#stripped} ))
            if [ "$colons" -eq 1 ]; then
                HOST="${raw%:*}"
                PORT="${raw##*:}"
            else
                HOST="$raw"
            fi
            ;;
        *)
            HOST="$raw"
            ;;
    esac

    case "$PORT" in
        ''|*[!0-9]*)
            echo "lb: invalid port in WEBUI_URL='$1'" >&2
            exit 1
            ;;
    esac
    if [ -z "$HOST" ]; then
        echo "lb: failed to parse host from WEBUI_URL='$1'" >&2
        exit 1
    fi

    origin_host="$HOST"
    case "$HOST" in
        *:*) origin_host="[${HOST}]" ;;
    esac
    if [ "$PORT" = "443" ]; then
        ORIGIN="https://${origin_host}"
        HTTPS_SUFFIX=""
    else
        ORIGIN="https://${origin_host}:${PORT}"
        HTTPS_SUFFIX=":${PORT}"
    fi
}

is_ip() {
    addr="$1"
    echo "$addr" | grep -Eq '^[0-9]{1,3}(\.[0-9]{1,3}){3}$' && return 0
    echo "$addr" | grep -Fq ':' && return 0
    return 1
}

link_canonical() {
    src="$1"
    dest="$2"
    if [ "$src" = "$dest" ]; then
        return 0
    fi
    ln -sfn "$src" "$dest"
}

generate_self_signed() {
    echo "lb: generating self-signed certificate for ${ORIGIN} (${TLS_DAYS} days)"

    if is_ip "$HOST"; then
        san="IP.1 = ${HOST}
IP.2 = 127.0.0.1
DNS.1 = localhost"
    elif [ "$HOST" = "localhost" ]; then
        san="DNS.1 = localhost
IP.1 = 127.0.0.1"
    else
        san="DNS.1 = ${HOST}
DNS.2 = localhost
IP.1 = 127.0.0.1"
    fi

    conf="$(mktemp)"
    cat > "$conf" <<EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = ${HOST}

[v3_req]
subjectAltName = @alt_names
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
basicConstraints = CA:false

[alt_names]
${san}
EOF

    openssl req -x509 -nodes -newkey rsa:2048 \
        -keyout "$KEY" \
        -out "$CERT" \
        -days "$TLS_DAYS" \
        -config "$conf" \
        -extensions v3_req
    chmod 644 "$CERT"
    chmod 600 "$KEY"
    rm -f "$conf"
}

parse_webui_url "$WEBUI_URL"
echo "lb: origin ${ORIGIN} (host=${HOST} port=${PORT})"

MARKER="${CERT_DIR}/.selfsigned-for"

if [ -s "$CERT" ] && [ -s "$KEY" ]; then
    if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" != "$ORIGIN" ]; then
        echo "lb: WEBUI_URL changed ($(cat "$MARKER") → ${ORIGIN}), regenerating"
        generate_self_signed
        printf '%s\n' "$ORIGIN" > "$MARKER"
    else
        echo "lb: using existing TLS files"
        echo "    cert: $CERT"
        echo "    key:  $KEY"
    fi
else
    generate_self_signed
    printf '%s\n' "$ORIGIN" > "$MARKER"
fi

link_canonical "$CERT" "$CANON_CERT"
link_canonical "$KEY" "$CANON_KEY"

sed -e "s|__SERVER_NAME__|${HOST}|g" \
    -e "s|__HTTPS_SUFFIX__|${HTTPS_SUFFIX}|g" \
    /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

nginx -t
exec nginx -g "daemon off;"
