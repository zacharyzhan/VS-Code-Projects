import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt

# 定义常量
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# 定义数据集路径（真实和虚假图像）
train_data_dir = '../input/real-vs-fake/train'
validation_data_dir = '../input/real-vs-fake/valid'

# 定义模型
model = keras.Sequential()
# 添加卷积层，使用32个3x3的卷积核，激活函数为ReLU，输入形状为指定的图像大小和颜色通道
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
# 添加最大池化层，池化窗口大小为2x2
model.add(layers.MaxPooling2D((2, 2)))
# 添加卷积层，使用64个3x3的卷积核，激活函数为ReLU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 添加最大池化层，池化窗口大小为2x2
model.add(layers.MaxPooling2D((2, 2)))
# 添加卷积层，使用128个3x3的卷积核，激活函数为ReLU
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# 添加最大池化层，池化窗口大小为2x2
model.add(layers.MaxPooling2D((2, 2)))
# 将多维输入一维化
model.add(layers.Flatten())
# 添加全连接层，包含128个神经元，激活函数为ReLU
model.add(layers.Dense(128, activation='relu'))
# 添加Dropout层，丢弃率为50%
model.add(layers.Dropout(0.5))
# 添加输出层，包含1个神经元，激活函数为Sigmoid
model.add(layers.Dense(1, activation='sigmoid'))  # 输出层，单神经元用于二元分类

# 编译模型
# 配置模型的优化器、损失函数和评估指标
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# 使用 ImageDataGenerator 准备数据
# 创建图像数据生成器，用于训练数据的归一化处理
train_datagen = ImageDataGenerator(rescale=1./255)
# 创建图像数据生成器，用于验证数据的归一化处理
validation_datagen = ImageDataGenerator(rescale=1./255)

# 从目录中生成训练数据的批处理生成器
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 从目录中生成验证数据的批处理生成器
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 在训练生成器上拟合模型并定义训练过程
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples #BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples #BATCH_SIZE
)

""" 
# 绘制模型结构
tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
# 打印模型结构
model.summary()
 """

# 绘制训练和验证的损失
plt.figure(figsize=(12, 4))

# 绘制损失
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失 (Loss)')
plt.xlabel('轮次 (Epochs)')
plt.ylabel('损失值 (Loss)')
plt.legend()

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('准确率 (Accuracy)')
plt.xlabel('轮次 (Epochs)')
plt.ylabel('准确率 (Accuracy)')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()

# 保存训练后的模型
model.save('./best_model.h5')