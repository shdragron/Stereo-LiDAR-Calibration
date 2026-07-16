"""Lazy relay: expose ONLY the race-relevant ZED topics on the public graph.

In race mode the ZED wrapper runs under a hidden namespace (`/_zed_hidden`,
leading underscore = not shown by `ros2 topic list`), and this node republishes
just left/right compressed images + camera_info under short public names:
/sensors/camera/{left,right}/{compressed,info}.

Lazy: each input is subscribed only while the public output has subscribers,
so the wrapper's JPEG encoding stays off until someone actually consumes it.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo

HIDDEN = '/_zed_hidden/zed'

# (type, hidden input suffix, short public name)
RELAYS = [
    (CompressedImage, '/left/color/rect/image/compressed',
     '/sensors/camera/left/compressed'),
    (CompressedImage, '/right/color/rect/image/compressed',
     '/sensors/camera/right/compressed'),
    (CameraInfo, '/left/color/rect/camera_info',
     '/sensors/camera/left/info'),
    (CameraInfo, '/right/color/rect/camera_info',
     '/sensors/camera/right/info'),
]

QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST, depth=10)


class ZedRelay(Node):
    def __init__(self):
        super().__init__('zed_relay')
        self._pairs = []           # (msg_type, in_topic, publisher, sub_or_None)
        for typ, suffix, out in RELAYS:
            pub = self.create_publisher(typ, out, QOS)
            self._pairs.append([typ, HIDDEN + suffix, pub, None])
        self.create_timer(1.0, self._manage)
        self.get_logger().info(
            f'lazy relay {HIDDEN}/* -> /sensors/camera/* ({len(self._pairs)} topics)')

    def _manage(self):
        for pair in self._pairs:
            typ, in_topic, pub, sub = pair
            wanted = pub.get_subscription_count() > 0
            if wanted and sub is None:
                pair[3] = self.create_subscription(
                    typ, in_topic, lambda m, p=pub: p.publish(m), QOS)
                self.get_logger().info(f'relay ON  {in_topic}')
            elif not wanted and sub is not None:
                self.destroy_subscription(sub)
                pair[3] = None
                self.get_logger().info(f'relay OFF {in_topic}')


def main(argv=None):
    rclpy.init(args=argv)
    node = ZedRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
