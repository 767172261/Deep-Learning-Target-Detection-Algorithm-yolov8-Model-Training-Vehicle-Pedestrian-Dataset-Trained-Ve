深度学习目标检测算法yolov8模型训练车辆行人数据集 通过训练出的车辆行人权重  基于深度学习yolov8模型的行人车辆检测系统
@[toc]
代码示例，仅供参考。

基于深度学习yolov8的行人车辆检测系统
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5baace80d60642b0b55fd27cf9de0ecc.png)
3176张行人车辆数据集
包含![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/520681acea8d4211860b2eb330cdedc1.png)


软件使用：
模型：YOLOV8
软件：Pycharm+Anaconda
环境：python=3.9 opencv_python PyQt5
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/37b5419fa3bf4c7694c99abc5db61d3b.png)


预实现功能目标： 对于多种角度的识别检测功能：包括通过选择图片、视频、摄像头进行实时识别
①选择图片识别行人和车辆类型。
②选择视频识别行人和车辆类型。
③摄像头检测识别。

基于深度学习的行人车辆检测系统，代码示例，仅供参考。
帮助你从数据集准备到最终系统的实现。

### 一、环境配置

确保你的开发环境已经安装了必要的软件和库：

```bash
# 安装Anaconda
# 下载并安装Anaconda: https://www.anaconda.com/products/distribution

# 创建并激活虚拟环境
conda create -n vehicle_detection python=3.9
conda activate vehicle_detection

# 安装必要的库
pip install ultralytics opencv-python PyQt5 pyqt5-tools
```

### 二、数据集准备

假设你同学已经有了包含3176张图片的数据集，并且已经标注好。数据集应该按照以下结构组织：

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

`data.yaml` 文件内容如下：

```yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 7
names: ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'person', 'train']
```

### 三、模型训练

使用YOLOv8进行模型训练：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8s.pt')

# 开始训练
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='vehicle_detection_train',
    project='runs/training',
    exist_ok=True,
    device=0 if torch.cuda.is_available() else 'cpu'
)

# 保存最佳权重
best_weights_path = model.save('weights/best_vehicle_detection.pt')
print(f"Best weights saved to {best_weights_path}")
```

### 四、推理代码

创建一个推理类 `VehicleDetector`：

```python
from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_path='weights/best_vehicle_detection.pt'):
        self.model = YOLO(model_path)

    def detect_image(self, image_path, save_path=None):
        results = self.model.predict(image_path, conf=0.25)
        for result in results:
            im_array = result.plot()
            im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
            if save_path:
                cv2.imwrite(save_path, im)
            cv2.imshow("Detection", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detect_video(self, video_path, save_path=None):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, conf=0.25)
            for result in results:
                annotated_frame = result.plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Frame', annotated_frame)
                if save_path:
                    out.write(annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if save_path:
            out.release()
        cv2.destroyAllWindows()

    def detect_webcam(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, conf=0.25)
            for result in results:
                annotated_frame = result.plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Webcam Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
```

### 五、UI界面设计

使用PyQt5设计UI界面：

```python
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from vehicle_detector import VehicleDetector

class VehicleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.detector = VehicleDetector()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("行人车辆检测系统")
        layout = QVBoxLayout()

        self.image_label = QLabel("请选择图片或视频")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        btn_image = QPushButton("选择图片并检测")
        btn_image.clicked.connect(self.upload_and_detect_image)
        layout.addWidget(btn_image)

        btn_video = QPushButton("选择视频并检测")
        btn_video.clicked.connect(self.upload_and_detect_video)
        layout.addWidget(btn_video)

        btn_webcam = QPushButton("打开摄像头检测")
        btn_webcam.clicked.connect(self.detect_webcam)
        layout.addWidget(btn_webcam)

        self.setLayout(layout)

    def upload_and_detect_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image files (*.jpg *.png)')
        if fname:
            result_img = 'runs/gui_result.jpg'
            self.detector.detect_image(fname, result_img)
            pixmap = QPixmap(result_img)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def upload_and_detect_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择视频', '', 'Video files (*.mp4 *.avi)')
        if fname:
            result_video = 'runs/gui_result.mp4'
            self.detector.detect_video(fname, result_video)

    def detect_webcam(self):
        self.detector.detect_webcam()

if __name__ == '__main__':
    app = QApplication([])
    window = VehicleApp()
    window.resize(800, 600)
    window.show()
    app.exec_()
```

### 六、运行效果

你可以通过以下命令运行UI界面：

```bash
python vehicle_app.py
```

启动一个图形界面，允许你选择图片、视频或打开摄像头进行实时检测。

使用PyQt5来设计和实现。以下是一个简单的示例代码，展示如何创建一个包含用户名、密码输入框以及登录按钮的登录界面，并在用户点击登录按钮时进行验证。
代码示例，仅供参考。

### 登录界面代码

```python
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('用户登录')
        layout = QVBoxLayout()

        # 用户名标签和输入框
        username_label = QLabel('用户名:')
        self.username_input = QLineEdit()
        layout.addWidget(username_label)
        layout.addWidget(self.username_input)

        # 密码标签和输入框
        password_label = QLabel('密码:')
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)  # 设置密码显示为星号
        layout.addWidget(password_label)
        layout.addWidget(self.password_input)

        # 登录按钮
        login_button = QPushButton('登录')
        login_button.clicked.connect(self.login)
        layout.addWidget(login_button)

        # 注册链接
        register_link = QLabel('<a href="#">还没有账号？点击注册</a>')
        register_link.setAlignment(Qt.AlignCenter)
        register_link.setOpenExternalLinks(True)
        layout.addWidget(register_link)

        self.setLayout(layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        # 这里只是一个简单的示例，实际应用中应该连接数据库进行验证
        if username == 'test' and password == 'password':
            QMessageBox.information(self, '登录成功', '欢迎回来！')
            # 在这里可以跳转到主界面或其他操作
        else:
            QMessageBox.warning(self, '登录失败', '用户名或密码错误，请重试。')

if __name__ == '__main__':
    app = QApplication([])
    window = LoginWindow()
    window.resize(300, 200)
    window.show()
    app.exec_()
```

### 说明

1. **用户名和密码输入框**：`QLineEdit` 用于输入用户名和密码。密码输入框通过 `setEchoMode(QLineEdit.Password)` 设置为星号显示。
2. **登录按钮**：当用户点击登录按钮时，会触发 `login` 方法进行验证。
3. **消息提示**：使用 `QMessageBox` 显示登录成功或失败的消息。
4. **注册链接**：使用 `QLabel` 和 `<a>` 标签创建一个可点击的注册链接，可以通过 `setOpenExternalLinks(True)` 打开外部链接。

### 登录界面集成到现有系统

登录界面集成到现有的行人车辆检测系统中，作为程序启动时的第一个窗口。例如，在 `VehicleApp` 类中添加一个方法来显示登录窗口，并在登录成功后显示主界面：

```python
class VehicleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.detector = VehicleDetector()
        self.login_window = LoginWindow()
        self.login_window.show()
        self.login_window.login_successful.connect(self.show_main_window)

    def show_main_window(self):
        self.login_window.hide()
        self.initUI()  # 初始化主界面
        self.show()

# 修改 LoginWindow 类以发送登录成功的信号
from PyQt5.QtCore import pyqtSignal

class LoginWindow(QWidget):
    login_successful = pyqtSignal()

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if username == 'test' and password == 'password':
            self.login_successful.emit()
            QMessageBox.information(self, '登录成功', '欢迎回来！')
        else:
            QMessageBox.warning(self, '登录失败', '用户名或密码错误，请重试。')
```


