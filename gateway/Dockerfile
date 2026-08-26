FROM node:20-slim

# python3 + build-essential: node-gyp fallback in case better-sqlite3 has no
# prebuilt binary for the host's platform/arch (it ships prebuilds for common
# linux x86_64/arm64 targets, so this is a safety net, not the common path).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY . .

# node:20-slim already ships a non-root "node" user at uid/gid 1000 -- the common
# default first-user id on Linux and macOS. Run as that instead of root: data/ is
# bind-mounted from the host (see docker-compose.yml, and scripts/tamper_suite.sh
# and seed_demo_data.sh, which edit certiproof.db directly by host path), and a
# root-owned container process would leave files there unwritable from the host.
RUN mkdir -p /app/data && chown -R node:node /app
USER node

ENV NODE_ENV=production
ENV DB_PATH=/app/data/certiproof.db

EXPOSE 4000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=10 \
  CMD node -e "require('http').get('http://127.0.0.1:4000/health', r => process.exit(r.statusCode===200?0:1)).on('error', () => process.exit(1))"

CMD ["node", "src/app.js"]
