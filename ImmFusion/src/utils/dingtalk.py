import ctypes
from multiprocessing import Value, Queue, Process
from time import sleep
import cv2
import base64
from dingtalkchatbot.chatbot import DingtalkChatbot


WEBHOOK_KEY = "https://oapi.dingtalk.com/robot/send?access_token=cb8414595ba360be066cc514e8541af1edc3f04ae1306f03b6b7325f2a01ddbd"
MSG_INFO = 1
MSG_ERROR = 2


class TimerBot(Process):
    """
    Kinect Subscriber
    """
    def __init__(self, interval: float = 10) -> None:
        super().__init__(daemon=True)
        self.info_queue = Queue(100)
        self.error_queue = Queue(100)
        self.md_queue = Queue(20)
        
        self.interval = Value(ctypes.c_float, interval)
        self.enabled = Value(ctypes.c_bool, False)

        self.start()
    
    def run(self):
        bot = DingtalkChatbot(WEBHOOK_KEY)

        while True:
            if not self.enabled.value:
                sleep(0.5)
                continue
            try:
                if not self.error_queue.empty():
                    text = self.error_queue.get()
                    bot.send_text("【ERROR】" + text, True)
                if not self.info_queue.empty():
                    text = self.info_queue.get()
                    bot.send_text("【INFO】" + text, False)
                if not self.md_queue.empty():
                    md = self.md_queue.get()
                    bot.send_markdown(**md)
            except Exception as e:
                print(e)
            # Sleep only if no error waits in queue
            if self.error_queue.empty():
                sleep(self.interval.value)

    def enable(self):
        self.enabled.value = True
    
    def disable(self):
        self.enabled.value = False

    def add_task(self, text: str, level: int = 1):
        if level == 1:
            self.keep_queue_available(self.info_queue)
            self.info_queue.put(text)
        else:
            self.keep_queue_available(self.error_queue)
            self.error_queue.put(text)

    def print(self, text):
        print(text)
        if self.enabled.value:
            self.add_task(text)

    def add_md(self, title: str, text: str, is_at_all: bool = False, at_mobiles: list = []):
        self.keep_queue_available(self.md_queue)
        self.md_queue.put(dict(
            title=title,
            text=text,
            is_at_all=is_at_all,
            at_mobiles=at_mobiles
        ))

    def img2b64(self, img, quality: int = 50):
        img_resized = cv2.resize(img, (256, 256))
        b64_str = ""
        while quality > 0:
            b64_str = cv2.imencode('.jpg', img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tostring()
            b64_str = base64.b64encode(b64_str).decode('utf-8')
            if len(b64_str) > 5500:
                quality -= 3
            else:
                break
        return "data:image/jpg;base64,{}".format(b64_str)

    def keep_queue_available(self, queue: Queue):
        if queue.full():
            queue.get()
            queue.get()
            queue.get()
