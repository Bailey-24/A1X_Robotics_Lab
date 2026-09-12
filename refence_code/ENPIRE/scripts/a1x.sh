#!/usr/bin/env bash
# Single-terminal launcher for the ENPIRE A1X integration.
#
# Usage:
#   scripts/a1x.sh install                     # one-time: sync the ENPIRE uv env
#   scripts/a1x.sh up [--allow-motion]         # start ROS + bridge + SAM3 in background
#   scripts/a1x.sh status                      # show bridge health + components
#   scripts/a1x.sh smoke                       # read-only state + camera test
#   scripts/a1x.sh run [--allow-motion] "<task>" [hydra overrides...]
#                                             # start stack, run agent, auto-stop
#   scripts/a1x.sh down                        # stop everything this script started
#
# Set A1X_START_SAM3=0 to skip the SAM3 segmentation server (GPU service).
#
# Examples:
#   scripts/a1x.sh up
#   scripts/a1x.sh run "使用腕部相机抓取红色方块" max_iterations=1
#   scripts/a1x.sh run --allow-motion "pick up the red cube" max_iterations=1
#
# Only processes started by this script are stopped by `down`; an A1X ROS
# stack you launched yourself is detected and left untouched.

set -euo pipefail

SDK_ROOT="${A1X_SDK_ROOT:-/home/ubuntu/projects/A1Xsdk}"
ENPIRE_ROOT="$SDK_ROOT/refence_code/ENPIRE"
RUN_DIR="${A1X_RUNTIME_DIR:-/tmp/a1x-enpire}"
ROS_PY="${A1X_ROS_PY:-/home/ubuntu/miniconda3/envs/a1x_ros/bin/python}"
SAM3_PY="${A1X_SAM3_PY:-/home/ubuntu/miniconda3/envs/sam3/bin/python}"
UV="${UV:-$HOME/.local/bin/uv}"
BRIDGE_PORT="${A1X_BRIDGE_PORT:-11337}"
BRIDGE_URL="${A1X_BRIDGE_URL:-http://127.0.0.1:$BRIDGE_PORT}"
SAM3_PORT="${A1X_SAM3_PORT:-9500}"
SAM3_URL="http://127.0.0.1:$SAM3_PORT"
START_SAM3="${A1X_START_SAM3:-1}"

# This station's D405 firmware is newer than ENPIRE's tested version.
export ENPIRE_ALLOW_D405_FIRMWARE="${ENPIRE_ALLOW_D405_FIRMWARE:-1}"

# VLM backend for task verification (GLM via the OpenAI-compatible "nvidia"
# backend: NVIDIA_API_KEY / NVIDIA_VL_BASE_URL / NVIDIA_VL_MODEL in ~/.zshrc).
export A1X_VLM_BACKEND="${A1X_VLM_BACKEND:-nvidia}"

# Table surface height in the A1X base frame (measured: grasp points on the
# 4 cm cube land at z≈-0.02, i.e. table ≈ -0.04). The bridge's Cartesian
# fence floor sits 2 mm above it so on-table grasps stay reachable while
# through-the-table motion stays blocked. Override with A1X_TABLE_Z_M.
A1X_TABLE_Z_M="${A1X_TABLE_Z_M:--0.04}"
export A1X_TABLE_Z_M
WS_Z_MIN=$(awk -v t="$A1X_TABLE_Z_M" 'BEGIN { printf "%.4f", t + 0.002 }')

# Keep localhost traffic off any configured HTTP proxy.
export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

sam3_health() {
  curl -sf --noproxy '*' --max-time 3 "$SAM3_URL/health" || true
}

log() { printf '[a1x] %s\n' "$*"; }
die() { printf '[a1x] ERROR: %s\n' "$*" >&2; exit 1; }

pid_alive() {
  [ -f "$RUN_DIR/$1.pid" ] && kill -0 "$(cat "$RUN_DIR/$1.pid")" 2>/dev/null
}

start_component() {
  local name=$1 logfile=$2
  shift 2
  if pid_alive "$name"; then
    log "$name already running (pid $(cat "$RUN_DIR/$name.pid"))"
    return 0
  fi
  mkdir -p "$RUN_DIR"
  setsid nohup "$@" >"$logfile" 2>&1 &
  local pid=$!
  echo "$pid" >"$RUN_DIR/$name.pid"
  log "$name started (pid $pid, log $logfile)"
}

stop_component() {
  local name=$1 pidfile="$RUN_DIR/$1.pid"
  if ! pid_alive "$name"; then
    rm -f "$pidfile"
    return 0
  fi
  local pid
  pid=$(cat "$pidfile")
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 25); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
  log "$name stopped"
}

external_running() {
  # The [x] char-class trick keeps the pattern from matching the caller's own
  # command line (pgrep -f scans full cmdlines, including this script's
  # invoker when the pattern appears in a shell command string).
  pgrep -f "$1" >/dev/null 2>&1
}

health() {
  curl -sf --max-time 3 "$BRIDGE_URL/health" || true
}

start_ros() { # name logfile launch-args...
  local name=$1 logfile=$2
  shift 2
  start_component "$name" "$logfile" \
    bash -c "source '$SDK_ROOT/install/setup.bash' && exec ros2 launch $*"
}

wait_ready() {
  local resp
  for _ in $(seq 1 60); do
    resp=$(health)
    if printf '%s' "$resp" | grep -q '"measured_joints": true' \
      && printf '%s' "$resp" | grep -q '"joint_tracker_connected": true'; then
      log "stack ready: $resp"
      return 0
    fi
    sleep 1
  done
  log "ERROR: stack not ready after 60s (last health: '$resp'); check logs in $RUN_DIR"
  return 1
}

cmd_install() {
  command -v "$UV" >/dev/null 2>&1 || die "uv not found at $UV (pip install --user uv)"
  cd "$ENPIRE_ROOT"
  "$UV" sync --extra dev --extra cap --extra vision --extra camera-realsense --extra vlm
  log "ENPIRE environment installed"
}

cmd_up() {
  local allow_motion="${1:-}"
  local bridge_args=()
  bridge_args+=(--workspace-min "0.05,-0.45,$WS_Z_MIN")
  if [ "$allow_motion" = "--allow-motion" ]; then
    bridge_args+=(--allow-motion)
    log "MOTION ENABLED — make sure the workspace is clear and you can reach the e-stop"
  fi

  if external_running "ros2 launch [H]DAS a1xy.py" || external_running "HDAS/lib/HDAS/[A]RM_APP"; then
    log "existing HDAS detected; not managed by this script"
  else
    start_ros hdas "$RUN_DIR/hdas.log" HDAS a1xy.py
  fi

  if external_running "[A]1x_jointTrackerdemo_launch.py" \
    || external_running "mobiman/a1_xy_jointTracker_[d]emo_node"; then
    log "existing joint tracker detected; not managed by this script"
  else
    start_ros tracker "$RUN_DIR/tracker.log" \
      mobiman A1x_jointTrackerdemo_launch.py launch_rviz:=false
  fi

  if external_running "[A]1xy_gripperController_launch.py"; then
    log "existing gripper controller detected; not managed by this script"
  else
    start_ros gripper "$RUN_DIR/gripper.log" mobiman A1xy_gripperController_launch.py
  fi

  if [ -n "$(health)" ]; then
    log "existing A1X bridge answering at $BRIDGE_URL; not managed by this script"
  else
    start_component bridge "$RUN_DIR/bridge.log" \
      bash -c "source '$SDK_ROOT/install/setup.bash' && exec '$ROS_PY' \
        '$ENPIRE_ROOT/enpire/env/forge/robot/a1x/bridge.py' \
        --port '$BRIDGE_PORT' ${bridge_args[*]+${bridge_args[*]}}"
  fi

  if [ "$START_SAM3" != "1" ]; then
    log "SAM3 server disabled (A1X_START_SAM3=0); segment_object will fail"
  elif [ -n "$(sam3_health)" ]; then
    log "existing SAM3 server answering at $SAM3_URL; not managed by this script"
  elif [ -x "$SAM3_PY" ]; then
    start_component sam3 "$RUN_DIR/sam3.log" \
      env PYTHONPATH="$ENPIRE_ROOT" \
        "$SAM3_PY" "$ENPIRE_ROOT/enpire/env/forge/tools/vision/serve_sam3.py" \
        --port "$SAM3_PORT" --preload
    local resp
    for _ in $(seq 1 30); do
      resp=$(sam3_health)
      [ -n "$resp" ] && break
      sleep 1
    done
    if [ -n "$resp" ]; then
      log "SAM3 ready: $resp"
    else
      log "WARNING: SAM3 server not healthy yet (loads on first segment call)"
    fi
  else
    log "WARNING: SAM3 python not found at $SAM3_PY; segment_object will fail"
  fi

  wait_ready
}

cmd_down() {
  stop_component sam3
  stop_component bridge
  stop_component gripper
  stop_component tracker
  stop_component hdas
  log "stack down"
}

cmd_status() {
  local resp
  for name in hdas tracker gripper bridge sam3; do
    if pid_alive "$name"; then
      log "$name: running (pid $(cat "$RUN_DIR/$name.pid"), managed)"
    elif [ "$name" = hdas ] && external_running "HDAS/lib/HDAS/[A]RM_APP"; then
      log "hdas: running (external)"
    elif [ "$name" = sam3 ] && [ -n "$(sam3_health)" ]; then
      log "sam3: running (external)"
    else
      log "$name: stopped"
    fi
  done
  resp=$(health)
  if [ -n "$resp" ]; then
    log "bridge health: $resp"
  else
    log "bridge health: unreachable at $BRIDGE_URL"
  fi
}

managed_before() {
  ls "$RUN_DIR"/*.pid 2>/dev/null || true
}

teardown_new() { # $1 = space-separated pidfiles that existed before `up`
  local before="$1" pidfile name
  for pidfile in "$RUN_DIR"/*.pid; do
    [ -e "$pidfile" ] || continue
    name=$(basename "$pidfile" .pid)
    if printf '%s' "$before" | grep -qF "$pidfile"; then
      continue
    fi
    stop_component "$name"
  done
}

cmd_smoke() {
  local before rc=0
  before=$(managed_before | tr '\n' ' ')
  if ! cmd_up; then
    teardown_new "$before"
    return 1
  fi
  cd "$ENPIRE_ROOT"
  "$UV" run --no-sync enpire-run-script \
    robot=real_a1x \
    task='A1X read-only smoke test' \
    script_file=cap/saved_scripts/a1x_smoke.py \
    robot.go_home_on_exit=false \
    runtime.quiet=false \
    runtime.exit_on_error=true || rc=$?
  teardown_new "$before"
  return "$rc"
}

cmd_run() {
  local allow_motion="" task="" extras=()
  while [ $# -gt 0 ]; do
    case "${1:-}" in
      --allow-motion)
        allow_motion="$1"
        shift
        ;;
      "")
        shift
        ;;
      *)
        if [ -z "$task" ]; then
          task="$1"
        else
          extras+=("$1")
        fi
        shift
        ;;
    esac
  done
  [ -n "$task" ] || die 'usage: a1x.sh run [--allow-motion] "<task>" [hydra overrides...]'

  local before rc=0
  before=$(managed_before | tr '\n' ' ')
  if ! cmd_up "$allow_motion"; then
    teardown_new "$before"
    return 1
  fi
  cd "$ENPIRE_ROOT"
  # Quote the task for Hydra's override grammar (handles spaces and non-ASCII).
  "$UV" run --no-sync enpire-run-agent \
    experiment=a1x_real \
    "task='$task'" \
    "${extras[@]}" || rc=$?
  teardown_new "$before"
  return "$rc"
}

case "${1:-help}" in
  install) shift; cmd_install "$@" ;;
  up) shift; cmd_up "${1:-}" ;;
  down) shift; cmd_down "$@" ;;
  status) shift; cmd_status "$@" ;;
  smoke) shift; cmd_smoke "$@" ;;
  run) shift; cmd_run "$@" ;;
  help|--help|-h)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    die "unknown command '$1' (try: up | down | status | smoke | run | install)"
    ;;
esac
