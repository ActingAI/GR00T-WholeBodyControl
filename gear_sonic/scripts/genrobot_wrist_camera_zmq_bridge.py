#!/usr/bin/env python3
from __future__ import annotations

"""Publish GENROBOT wrist ROS image topics as a SONIC camera ZMQ stream.

Run this on the G1 onboard computer after launching the GENROBOT ROS driver.
The workstation data exporter can subscribe to this as a secondary camera
source and merge the ``left_wrist`` / ``right_wrist`` images with ``ego_view``.
"""

import argparse
import threading
import time

import cv2
import msgpack
import numpy as np
import rospy
from sensor_msgs.msg import Image
import zmq


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    """Convert a ROS sensor_msgs/Image into a BGR OpenCV array."""
    encoding = msg.encoding.lower()
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)

    if encoding in ("bgr8", "rgb8"):
        image = row[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    if encoding in ("bgra8", "rgba8"):
        image = row[:, : msg.width * 4].reshape(msg.height, msg.width, 4)
        code = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(image, code)

    if encoding in ("mono8", "8uc1"):
        image = row[:, : msg.width].reshape(msg.height, msg.width)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if encoding in ("yuyv", "yuyv422", "yuv422"):
        image = row[:, : msg.width * 2].reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)

    raise ValueError(f"Unsupported ROS image encoding: {msg.encoding}")


class LatestJpegStore:
    def __init__(self, jpeg_quality: int):
        self._jpeg_quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._images: dict[str, bytes] = {}
        self._timestamps: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, key: str, msg: Image) -> None:
        try:
            image = ros_image_to_bgr(msg)
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
            if not ok:
                rospy.logwarn_throttle(1.0, "Failed to JPEG-encode %s", key)
                return
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "Failed to convert %s image: %s", key, exc)
            return

        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        if stamp <= 0.0:
            stamp = time.time()

        with self._lock:
            self._images[key] = encoded.tobytes()
            self._timestamps[key] = stamp
            self._counts[key] = self._counts.get(key, 0) + 1

    def snapshot(self) -> tuple[dict[str, float], dict[str, bytes], dict[str, int]]:
        with self._lock:
            return dict(self._timestamps), dict(self._images), dict(self._counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pub-port", type=int, default=5559, help="ZMQ PUB port")
    parser.add_argument("--fps", type=float, default=30.0, help="Publish rate")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality")
    parser.add_argument(
        "--left-topic",
        default="/left_gripper/camera/color/image_raw",
        help="Left GENROBOT central camera ROS topic",
    )
    parser.add_argument(
        "--right-topic",
        default="/right_gripper/camera/color/image_raw",
        help="Right GENROBOT central camera ROS topic",
    )
    args = parser.parse_args()

    rospy.init_node("genrobot_wrist_camera_zmq_bridge")
    store = LatestJpegStore(jpeg_quality=args.jpeg_quality)

    rospy.Subscriber(args.left_topic, Image, lambda msg: store.update("left_wrist", msg), queue_size=1)
    rospy.Subscriber(args.right_topic, Image, lambda msg: store.update("right_wrist", msg), queue_size=1)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 20)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{args.pub_port}")

    rospy.loginfo(
        "GENROBOT wrist camera bridge publishing tcp://*:%d from %s and %s",
        args.pub_port,
        args.left_topic,
        args.right_topic,
    )

    rate = rospy.Rate(args.fps)
    sent = 0
    while not rospy.is_shutdown():
        timestamps, images, counts = store.snapshot()
        if images:
            payload = {"timestamps": timestamps, "images": images}
            try:
                socket.send(msgpack.packb(payload, use_bin_type=True), flags=zmq.NOBLOCK)
                sent += 1
                if sent % 100 == 0:
                    rospy.loginfo(
                        "Published %d wrist camera messages; image keys=%s counts=%s",
                        sent,
                        sorted(images.keys()),
                        counts,
                    )
            except zmq.Again:
                pass
        rate.sleep()

    socket.close()
    context.term()


if __name__ == "__main__":
    main()
