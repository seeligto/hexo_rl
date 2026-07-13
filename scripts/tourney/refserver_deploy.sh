#!/usr/bin/env bash
# Idempotent deploy of hexo-ref-server (React arena/viz) to a vast.ai box.
# Usage:  bash refserver_deploy.sh
# Operator-run; safe to re-run (kills old tmux session, rebuilds, restarts).

set -euo pipefail

# ── Parameterize at top ────────────────────────────────────────────────────────
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vast_hexo}"
SSH_PORT="${SSH_PORT:-13523}"
SSH_HOST="${SSH_HOST:-root@ssh7.vast.ai}"
REMOTE_DIR="/workspace/hexo-ref-server"
LOCAL_SRC="${LOCAL_SRC:-$HOME/Work/Hexo/hexo-ref-server}"
BIND_PORT="${BIND_PORT:-8080}"
TMUX_SESSION="refserver"

SSH_OPTS="-i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"

remote() { ssh $SSH_OPTS "$SSH_HOST" "$@"; }

echo "[refserver_deploy] === Step 1: Install node 22 + pnpm 10.20.0 ==="
remote 'bash -s' << 'ENDSSH'
export NVM_DIR="$HOME/.nvm"
if [ ! -d "$NVM_DIR" ]; then
  curl -fsSo- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi
. "$NVM_DIR/nvm.sh"
nvm install 22 2>/dev/null || true
node --version
corepack enable
corepack prepare pnpm@10.20.0 --activate 2>/dev/null || true
pnpm --version
ENDSSH

echo "[refserver_deploy] === Step 2: rsync source ==="
rsync -a --delete \
  -e "ssh $SSH_OPTS" \
  --exclude='node_modules' \
  --exclude='.git' \
  --exclude='packages/*/dist' \
  --exclude='packages/*/node_modules' \
  --exclude='logs' \
  --exclude='packages/backend/.env' \
  "$LOCAL_SRC/" \
  "$SSH_HOST:$REMOTE_DIR/"

echo "[refserver_deploy] === Step 3: pnpm install + build ==="
remote 'bash -s' << ENDSSH
export NVM_DIR="\$HOME/.nvm"
. "\$NVM_DIR/nvm.sh"
cd $REMOTE_DIR

# Allow build scripts for native deps
cat > .npmrc << 'EOF'
onlyBuiltDependencies[]=esbuild
onlyBuiltDependencies[]=mongodb-memory-server
EOF

pnpm install --frozen-lockfile
# esbuild needs an explicit rebuild after pnpm install in security mode
pnpm rebuild esbuild 2>/dev/null || true
pnpm rebuild mongodb-memory-server 2>/dev/null || true
pnpm build
ENDSSH

echo "[refserver_deploy] === Step 4: Write .env ==="
remote "bash -c 'mkdir -p $REMOTE_DIR/packages/backend && cat > $REMOTE_DIR/packages/backend/.env' " << ENVEOF
NODE_ENV=development
PORT=$BIND_PORT
MONGODB_USE_MEMORY=true
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DB_NAME=ih3t
AUTH_SECRET=tourney-local-secret
DISCORD_CLIENT_ID=tourney-unused
DISCORD_CLIENT_SECRET=tourney-unused
FRONTEND_DIST_PATH=$REMOTE_DIR/packages/frontend/dist/
ENVEOF

echo "[refserver_deploy] === Step 5: Free port $BIND_PORT if blocked by jupyter ==="
remote "bash -c 'PID=\$(ss -tlnp | grep \":$BIND_PORT\" | grep -oP \"pid=\K[0-9]+\" | head -1); [ -n \"\$PID\" ] && kill \$PID && sleep 2 && echo \"Killed PID \$PID\" || echo \"Port $BIND_PORT already free\"'"

echo "[refserver_deploy] === Step 6: Launch in tmux ==="
remote 'bash -s' << ENDSSH
export NVM_DIR="\$HOME/.nvm"
. "\$NVM_DIR/nvm.sh"
NODE_BIN="\$(which node)"

# Kill any existing session
tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
pkill -f "packages/backend/dist/server.cjs" 2>/dev/null || true
sleep 1

mkdir -p $REMOTE_DIR/logs
> $REMOTE_DIR/logs/server.log

tmux new-session -d -s $TMUX_SESSION \
  "\$NODE_BIN $REMOTE_DIR/packages/backend/dist/server.cjs 2>&1 | tee -a $REMOTE_DIR/logs/server.log"
echo "tmux session launched"
ENDSSH

echo "[refserver_deploy] === Step 7: Wait for :$BIND_PORT to respond ==="
remote "until curl -sS -o /dev/null -w '%{http_code}' http://localhost:$BIND_PORT/ 2>/dev/null | grep -qE '^[23]'; do sleep 5; done; echo 'SERVER_UP'"

echo "[refserver_deploy] === Step 8: Verify ==="
STATUS=$(remote "curl -sS -o /dev/null -w '%{http_code}' http://localhost:$BIND_PORT/")
echo "HTTP status: $STATUS"
remote "curl -s http://localhost:$BIND_PORT/ | head -3"

echo ""
echo "[refserver_deploy] DONE. Server live at box:$BIND_PORT"
echo "SSH tunnel (run locally):"
echo "  ssh -i $SSH_KEY -p $SSH_PORT -N -L $BIND_PORT:localhost:$BIND_PORT $SSH_HOST"
echo "Then open: http://localhost:$BIND_PORT"
echo ""
echo "Restart: tmux attach -t $TMUX_SESSION  (on box)"
echo "Logs:    tail -f $REMOTE_DIR/logs/server.log"
