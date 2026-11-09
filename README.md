# 眼底血管分割

这个项目使用U-Net对眼底图像进行血管分割，训练和预测在Kaggle Notebook中完成。代码在原图分辨率上训练，使用三折交叉验证，预测时平均三个模型的输出。

## 代码文件

| 文件 | 用途 |
| --- | --- |
| `cell_01_data.py` | 配置训练参数，读取数据并预处理图像 |
| `cell_02_model.py` | 定义模型、损失函数和滑窗预测 |
| `cell_03_train.py` | 进行三折训练，保存各折最佳权重并选择分割阈值 |
| `cell_04_predict_submit.py` | 汇总三个模型的预测，生成并检查提交文件 |

## 数据集

使用Kaggle任务提供的眼底血管分割数据。`train/image/`包含20张训练图像，`train/label/`包含对应的血管标注，`test/image/`包含20张测试图像。数据根目录还需包含`sample_submission.csv`和`segmentation_to_csv.py`。

## 在Kaggle上运行

1. 在Kaggle Notebook中添加原始数据并启用GPU。若代码无法自动定位数据，在`cell_01_data.py`中将`DATA_ROOT`设为实际挂载的`segmentation`目录。
2. 确认环境包含PyTorch、NumPy、pandas、OpenCV、Pillow和IPython。
3. 将4个`.py`文件分别放进同一个Notebook的4个代码单元，按`01`到`04`依次运行。后面的单元会使用前面定义的变量和函数。
4. 提交文件为`/kaggle/working/submission.csv`。训练权重、配置和黑色血管预测图保存在`/kaggle/working/vessel_native_3models/`。
