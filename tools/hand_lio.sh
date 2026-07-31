#!/usr/bin/env bash
# hand_lio 启停脚本
#
# 用法: hand_lio.sh {start|stop|restart|status|log} [roslaunch 额外参数...]
#
# - start 时自动准备环境:roslaunch 不在 PATH 且有 mamba 时激活 ros_host,
#   再 source 工作区 devel/setup.bash(工作区默认取脚本位置向上三级,可用
#   环境变量 ROS_WS 覆盖,如设备上路径不同时)。
# - ROS master 不可达时自动后台起一个 roscore,pid 记在 <ws>/run/roscore.pid。
#   roscore 由各包脚本共用,stop 不会杀它;不需要时 kill $(cat run/roscore.pid)。
# - stop 先发 SIGINT 让 roslaunch 走正常关闭流程,15s 不退再升级 TERM/KILL。
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS_DIR=${ROS_WS:-$(cd "$SCRIPT_DIR/../../.." && pwd)}

NAME=hand_lio
LAUNCH_PKG=hand_lio
LAUNCH_FILE=hand_lio.launch
DEFAULT_ARGS=()

RUN_DIR=$WS_DIR/run
PID_FILE=$RUN_DIR/$NAME.pid
LOG_FILE=$RUN_DIR/$NAME.log
ROSCORE_PID_FILE=$RUN_DIR/roscore.pid
ROSCORE_LOG_FILE=$RUN_DIR/roscore.log

pid_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

setup_env() {
    if ! command -v roslaunch >/dev/null 2>&1 && command -v mamba >/dev/null 2>&1; then
        # 开发机:ROS 装在 mamba 的 ros_host 环境里;设备上系统自带 ROS 则不会走到这
        eval "$(mamba shell hook --shell bash)"
        mamba activate ros_host
    fi
    if [ ! -f "$WS_DIR/devel/setup.bash" ]; then
        echo "找不到 $WS_DIR/devel/setup.bash,先 catkin_make,或用 ROS_WS 指定工作区" >&2
        exit 1
    fi
    source "$WS_DIR/devel/setup.bash"
}

ensure_master() {
    if timeout 3 rostopic list >/dev/null 2>&1; then return; fi
    if pid_alive "$ROSCORE_PID_FILE"; then return; fi # roscore 刚起还没就绪
    echo "ROS master 不可达,后台启动 roscore(日志: $ROSCORE_LOG_FILE)"
    nohup roscore >"$ROSCORE_LOG_FILE" 2>&1 &
    echo $! >"$ROSCORE_PID_FILE"
}

start() {
    if pid_alive "$PID_FILE"; then
        echo "$NAME 已在运行 (pid $(cat "$PID_FILE"))"
        return
    fi
    setup_env
    mkdir -p "$RUN_DIR"
    ensure_master
    nohup roslaunch --wait "$LAUNCH_PKG" "$LAUNCH_FILE" "${DEFAULT_ARGS[@]}" "$@" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    sleep 3
    if pid_alive "$PID_FILE"; then
        echo "$NAME 已启动 (pid $(cat "$PID_FILE")),日志: $LOG_FILE"
    else
        echo "$NAME 启动失败,日志尾部:" >&2
        tail -n 20 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop() {
    if ! pid_alive "$PID_FILE"; then
        echo "$NAME 未在运行"
        rm -f "$PID_FILE"
        return
    fi
    local pid
    pid=$(cat "$PID_FILE")
    echo "停止 $NAME (pid $pid) ..."
    kill -INT "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "SIGINT 15s 未退出,升级 SIGTERM/SIGKILL"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 3
        kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "$NAME 已停止"
}

status() {
    if pid_alive "$PID_FILE"; then
        echo "$NAME 运行中 (pid $(cat "$PID_FILE"))"
    else
        echo "$NAME 未运行"
    fi
}

cmd=${1:-}
shift 2>/dev/null || true
case "$cmd" in
    start) start "$@" ;;
    stop) stop ;;
    restart)
        stop
        start "$@"
        ;;
    status) status ;;
    log) exec tail -n 50 -f "$LOG_FILE" ;;
    *)
        echo "用法: $(basename "$0") {start|stop|restart|status|log} [roslaunch 额外参数...]" >&2
        exit 1
        ;;
esac
