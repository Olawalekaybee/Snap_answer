#!/usr/bin/env bash
# Turns SnapAnswer into an "app" on the DGX Spark:
#   - the dashboard runs in the background and starts automatically when you log in
#   - a SnapAnswer icon (app menu + desktop) opens it in its own window
# Usage (once, from the snapanswer folder):   bash install_app.sh
# Remove again:                               bash install_app.sh --uninstall
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/snapanswer-venv}"
PORT="${PORT:-8000}"
UNIT_DIR="$HOME/.config/systemd/user"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"

if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl --user disable --now snapanswer.service 2>/dev/null || true
  rm -f "$UNIT_DIR/snapanswer.service" "$APPS_DIR/snapanswer.desktop" "$DESKTOP_DIR/snapanswer.desktop"
  systemctl --user daemon-reload
  echo "SnapAnswer app removed (your project files are untouched)."
  exit 0
fi

[[ -x "$VENV/bin/python" ]] || { echo "Python venv not found at $VENV - run setup_spark.sh first."; exit 1; }

echo "== Background service (starts automatically when you log in)"
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/snapanswer.service" <<EOF
[Unit]
Description=SnapAnswer dashboard (RealSense + AI answers)
After=network-online.target sound.target

[Service]
WorkingDirectory=$APP_DIR
ExecStart=$VENV/bin/python $APP_DIR/dashboard.py --port $PORT
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now snapanswer.service

echo "== Launcher script"
cat > "$APP_DIR/open_snapanswer.sh" <<EOF
#!/usr/bin/env bash
# Opens the SnapAnswer dashboard window (starts the background service if needed).
URL="http://localhost:$PORT"
systemctl --user start snapanswer.service
for i in \$(seq 1 60); do curl -s -o /dev/null "\$URL/state" && break; sleep 1; done
for b in google-chrome google-chrome-stable chromium chromium-browser microsoft-edge brave-browser; do
  if command -v "\$b" >/dev/null; then exec "\$b" --app="\$URL" --class=SnapAnswer; fi
done
exec xdg-open "\$URL"
EOF
chmod +x "$APP_DIR/open_snapanswer.sh"

echo "== Icon"
cat > "$APP_DIR/web/icon.svg" <<'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
  <rect width="128" height="128" rx="28" fill="#0d1117"/>
  <rect x="18" y="38" width="92" height="62" rx="12" fill="#2f81f7"/>
  <rect x="44" y="28" width="40" height="16" rx="6" fill="#2f81f7"/>
  <circle cx="64" cy="69" r="21" fill="#0d1117"/>
  <text x="64" y="79" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="28" font-weight="900" fill="#3fb950">A</text>
</svg>
EOF

echo "== App menu + desktop icon"
mkdir -p "$APPS_DIR"
cat > "$APPS_DIR/snapanswer.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SnapAnswer
Comment=Point the camera at a question and hear the answer
Exec=$APP_DIR/open_snapanswer.sh
Icon=$APP_DIR/web/icon.svg
Terminal=false
Categories=Education;Utility;
StartupWMClass=SnapAnswer
EOF
if [[ -d "$DESKTOP_DIR" ]]; then
  cp "$APPS_DIR/snapanswer.desktop" "$DESKTOP_DIR/snapanswer.desktop"
  chmod +x "$DESKTOP_DIR/snapanswer.desktop"
  gio set "$DESKTOP_DIR/snapanswer.desktop" metadata::trusted true 2>/dev/null || true
fi
update-desktop-database "$APPS_DIR" 2>/dev/null || true

sleep 3
echo
systemctl --user --no-pager --lines=5 status snapanswer.service || true
echo
echo "Done. Open 'SnapAnswer' from the app menu or the desktop icon."
echo "From your phone/laptop on the same Wi-Fi:  http://$(hostname -I | awk '{print $1}'):$PORT"
echo "Logs:     journalctl --user -u snapanswer -f"
echo "Restart:  systemctl --user restart snapanswer"