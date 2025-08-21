import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class NumberPublisher(Node):
    def __init__(self):
        super().__init__('number_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'numbers', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = [1.1, 2.2, 3.3]  # your list of numbers
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = NumberPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
