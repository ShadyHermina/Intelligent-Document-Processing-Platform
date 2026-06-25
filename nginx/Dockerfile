# ── Base image ────────────────────────────────────────────────────
# nginx:1.27-alpine uses Alpine Linux rather than Debian.
# We make an exception to the "avoid Alpine" rule here because:
# - Nginx has no Python dependencies, so musl libc is not a problem
# - The official nginx:alpine image is battle-tested and widely used
# - It is significantly smaller than nginx:1.27 (Debian-based)
# - curl installs cleanly on Alpine via apk with no compatibility issues
FROM nginx:1.27-alpine

# ── System dependencies ───────────────────────────────────────────
# Alpine uses apk instead of apt-get as its package manager.
# apk add --no-cache installs without keeping the package index cache.
# --no-cache on apk is equivalent to the apt-get update &&
# rm -rf /var/lib/apt/lists/* pattern we used in other Dockerfiles —
# it avoids storing the index in the image layer.
RUN apk add --no-cache curl

# ── Nginx config ──────────────────────────────────────────────────
# We copy our custom nginx.conf over the default config file.
# /etc/nginx/nginx.conf is the path Nginx reads on startup.
# Replacing it entirely rather than using the conf.d include pattern
# gives us full explicit control — nothing inherited from defaults
# that we didn't write and understand ourselves.
COPY nginx.conf /etc/nginx/nginx.conf

# ── Expose ───────────────────────────────────────────────────────
# Nginx listens on port 80 (standard HTTP).
# We do not use 443 (HTTPS) in Phase 0 — TLS termination is a
# later-phase concern. For a local dev environment, plain HTTP is fine.
EXPOSE 80