#!/usr/bin/env bash
# 운영 실행: 화면(site/dist)과 API를 한 포트로 띄우고, 죽으면 5초 뒤 다시 띄운다.
# 설정은 site/.env (SITE_ADMIN_PASSWORD, SESSION_SECRET, PORT). Cloudflare Tunnel은 http://127.0.0.1:$PORT 를 가리킨다.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/node22/bin:$PATH"
set -a; [ -f .env ] && . ./.env; set +a
export SERVE_WEB=1 NODE_ENV=production
mkdir -p logs
while true; do
  node node_modules/.bin/tsx server/index.ts >> logs/server.log 2>&1
  echo "$(date '+%F %T') server exited ($?), restarting in 5s" >> logs/server.log
  sleep 5
done
