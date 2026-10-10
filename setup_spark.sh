#!/usr/bin/env bash
# SnapAnswer one-time setup for NVIDIA DGX Spark (DGX OS / Ubuntu, ARM64).
# Also works on other Ubuntu 22.04/24.04 machines (x86_64 or ARM64).
#
# Usage:
#   bash setup_spark.sh                        # standard install (USB webcam such as the EMEET C60E)
#   WITH_REALSENSE=1 bash setup_spark.sh       # also build the Intel RealSense driver (D435/D435i)
#   WITH_BIG_MODEL=1 bash setup_spark.sh       # also download gpt-oss:20b for stronger offline answers
set -euo pipefail

VENV="${VENV:-$HOME/snapanswer-venv}"
WITH_REALSENSE="${WITH_REALSENSE:-0}"
WITH_BIG_MODEL="${WITH_BIG_MODEL:-0}"
LRS_BRANCH="${LRS_BRANCH:-master}"
HERE="$(cd "$(dirname "$0")" && pwd)"

step() { echo; echo "==> $*"; }

step "System packages"
sudo apt-get update
sudo apt-get install -y git curl python3-dev python3-venv espeak-ng libportaudio2 alsa-utils v4l-utils

step "Python environment at $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

if [[ "$WITH_REALSENSE" == "1" ]]; then
  step "Intel RealSense driver + Python bindings (built from source, takes 10 to 20 minutes)"
  sudo apt-get install -y cmake build-essential pkg-config libssl-dev libusb-1.0-0-dev libudev-dev \
    libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
  cd /tmp
  [ -d librealsense/.git ] || git clone --depth 1 --branch "$LRS_BRANCH" https://github.com/IntelRealSense/librealsense.git
  cd librealsense
  sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
  rm -rf build && mkdir build && cd build
  # RSUSB backend: userspace USB, so no kernel patching is needed.
  # Examples off: they need OpenGL, which fails to configure on DGX OS, and SnapAnswer doesn't use them.
  # Python pinned to the venv so CMake doesn't pick up another Python (PlatformIO, conda, ...).
  cmake .. -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$VENV/bin/python" -DPython_EXECUTABLE="$VENV/bin/python" \
    -DPython3_EXECUTABLE="$VENV/bin/python" \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=ON \
    -DBUILD_UNIT_TESTS=OFF -DBUILD_WITH_CUDA=OFF
  make -j"$(nproc)"
  sudo make install
  sudo ldconfig
  SITE="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  find . \( -name 'pyrealsense2*.so*' -o -name 'pybackend2*.so*' \) -exec cp -L {} "$SITE/" \;
  "$VENV/bin/python" -c "import pyrealsense2 as rs; print('pyrealsense2 OK, devices:', len(rs.context().devices))"
  cd "$HERE"
else
  echo "(Skipping the RealSense driver. Run with WITH_REALSENSE=1 if you use a D435/D435i.)"
fi

step "Ollama + offline vision model"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:8b
if [[ "$WITH_BIG_MODEL" == "1" ]]; then
  ollama pull gpt-oss:20b
fi

step "Kokoro voice (downloads about 330 MB the first time)"
"$VENV/bin/python" -c "from kokoro import KPipeline; KPipeline(lang_code='b', repo_id='hexgrad/Kokoro-82M'); print('Kokoro OK')"

step "Camera check"
v4l2-ctl --list-devices 2>/dev/null || echo "No webcam listed yet. Plug it in and run: v4l2-ctl --list-devices"

cat <<EOF

Setup finished.

Next steps:
  1. (Optional, for Online mode) save your Claude API key:
       echo "sk-ant-your-key" > "$HERE/.anthropic_key" && chmod 600 "$HERE/.anthropic_key"
  2. Try it once in the terminal:
       source "$VENV/bin/activate" && cd "$HERE" && python dashboard.py
     then open http://localhost:8000
  3. Install it as an app that starts automatically:
       bash "$HERE/install_app.sh"
EOF