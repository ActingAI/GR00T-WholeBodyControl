#!/usr/bin/env python3
"""Bridge SONIC Pico UDP gripper commands into GENROBOT DAS ROS topics.

Run this on the G1 onboard computer after sourcing the GENROBOT SDK ROS
workspace. The Pico manager on the workstation sends JSON UDP packets with
left/right target distances; this node republishes them as std_msgs/Float32.
"""

import argparse
import json
import socket
import time

import msgpack
import rospy
from std_msgs.msg import Float32
import zmq


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-host", default="0.0.0.0", help="UDP bind host")
    parser.add_argument("--bind-port", type=int, default=5568, help="UDP bind port")
    parser.add_argument(
        "--left-topic",
        default="/left_gripper/target_distance",
        help="ROS topic for the left GENROBOT target distance",
    )
    parser.add_argument(
        "--right-topic",
        default="/right_gripper/target_distance",
        help="ROS topic for the right GENROBOT target distance",
    )
    parser.add_argument("--min-distance", type=float, default=0.0, help="Minimum allowed distance")
    parser.add_argument("--max-distance", type=float, default=0.103, help="Maximum allowed distance")
    parser.add_argument(
        "--left-encoder-topic",
        default="/left_gripper/encoder",
        help="ROS topic for the left GENROBOT actual opening encoder",
    )
    parser.add_argument(
        "--right-encoder-topic",
        default="/right_gripper/encoder",
        help="ROS topic for the right GENROBOT actual opening encoder",
    )
    parser.add_argument(
        "--state-pub-port",
        type=int,
        default=5569,
        help="ZMQ PUB port for GENROBOT actual/target opening state",
    )
    parser.add_argument("--state-pub-rate", type=float, default=50.0, help="State publish rate in Hz")
    args = parser.parse_args()

    rospy.init_node("genrobot_gripper_udp_bridge")
    left_pub = rospy.Publisher(args.left_topic, Float32, queue_size=1)
    right_pub = rospy.Publisher(args.right_topic, Float32, queue_size=1)

    state = {
        "left_encoder": None,
        "right_encoder": None,
        "left_target": None,
        "right_target": None,
        "left_encoder_ros_time": 0.0,
        "right_encoder_ros_time": 0.0,
        "receive_timestamp": 0.0,
    }

    def _encoder_callback(side: str, msg: Float32):
        value = float(msg.data)
        now = time.time()
        state[f"{side}_encoder"] = value
        state[f"{side}_encoder_ros_time"] = rospy.get_time()
        state["receive_timestamp"] = now

    rospy.Subscriber(args.left_encoder_topic, Float32, lambda msg: _encoder_callback("left", msg))
    rospy.Subscriber(args.right_encoder_topic, Float32, lambda msg: _encoder_callback("right", msg))

    zmq_context = zmq.Context()
    state_socket = zmq_context.socket(zmq.PUB)
    state_socket.setsockopt(zmq.SNDHWM, 20)
    state_socket.setsockopt(zmq.LINGER, 0)
    state_socket.bind(f"tcp://*:{args.state_pub_port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind_host, args.bind_port))
    sock.settimeout(min(0.02, 1.0 / max(args.state_pub_rate, 1.0)))

    rospy.loginfo(
        "GENROBOT gripper UDP bridge listening on %s:%d -> %s, %s; state on tcp://*:%d",
        args.bind_host,
        args.bind_port,
        args.left_topic,
        args.right_topic,
        args.state_pub_port,
    )

    last_left = None
    last_right = None
    state_period = 1.0 / max(args.state_pub_rate, 1.0)
    next_state_publish = time.monotonic()
    while not rospy.is_shutdown():
        now_mono = time.monotonic()
        if now_mono >= next_state_publish:
            state_msg = {
                "timestamp": time.time(),
                "left_encoder": state["left_encoder"],
                "right_encoder": state["right_encoder"],
                "left_target": state["left_target"],
                "right_target": state["right_target"],
                "left_encoder_ros_time": state["left_encoder_ros_time"],
                "right_encoder_ros_time": state["right_encoder_ros_time"],
                "receive_timestamp": state["receive_timestamp"],
            }
            try:
                state_socket.send(msgpack.packb(state_msg, use_bin_type=True), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            next_state_publish = now_mono + state_period

        try:
            packet, address = sock.recvfrom(4096)
        except socket.timeout:
            continue

        try:
            msg = json.loads(packet.decode("utf-8"))
            left = clamp(msg["left"], args.min_distance, args.max_distance)
            right = clamp(msg["right"], args.min_distance, args.max_distance)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rospy.logwarn_throttle(1.0, "Ignoring malformed gripper UDP packet: %s", exc)
            continue

        left_pub.publish(Float32(data=left))
        right_pub.publish(Float32(data=right))
        state["left_target"] = left
        state["right_target"] = right
        if left != last_left or right != last_right:
            rospy.loginfo(
                "GENROBOT target distance left=%.4f right=%.4f from %s:%d",
                left,
                right,
                address[0],
                address[1],
            )
            last_left = left
            last_right = right


if __name__ == "__main__":
    main()
