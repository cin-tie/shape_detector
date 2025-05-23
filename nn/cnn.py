import numpy as np
from nn.layers import Conv2D, MaxPool2D, Flatten, Dense
from nn.activations import ReLU, Softmax
from nn.loss import cross_entropy_loss

SHAPES = ['circle', 'square', 'triangle', 'star', 'hexagon', 'pentagon']
COLORS = {'red':0, 'green':1, 'blue':2}

class CNN:
    def __init__(self):
        self.conv1 = Conv2D(3, 8, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2,2)
        self.conv2 = Conv2D(8, 16, 3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2,2)
        self.flatten = Flatten()
        self.fc1 = Dense(input_size=4096, output_size=64)
        self.relu3 = ReLU()
        self.fc_shape = Dense(64, len(SHAPES))
        self.fc_color = Dense(64, len(COLORS))

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = x.reshape(x.shape[0], -1)

        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)

        out_shape = self.fc_shape.forward(x)
        out_color = self.fc_color.forward(x)

        return out_shape, out_color

    def predict(self, x):
        # x shape: (H,W,3), нормализуем и меняем порядок на (1,C,H,W)
        x = x.transpose(2,0,1)[None, ...] / 255.0
        out_shape, out_color = self.forward(x)
        shape_probs = self.softmax(out_shape[0])
        color_probs = self.softmax(out_color[0])
        shape_idx = np.argmax(shape_probs)
        color_idx = np.argmax(color_probs)
        confidence = max(shape_probs[shape_idx], color_probs[color_idx]) * 100
        return {'shape': SHAPES[shape_idx], 'color': list(COLORS.keys())[color_idx], 'confidence': confidence}

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def train_batch(self, x_batch, y_shape_batch, y_color_batch, learning_rate=0.001):
        batch_size = x_batch.shape[0]
        loss = 0
        for i in range(batch_size):
            x = x_batch[i:i+1].transpose(0,3,1,2) / 255.0
            y_shape_true = y_shape_batch[i]
            y_color_true = y_color_batch[i]

            out_shape, out_color = self.forward(x)

            shape_loss, grad_shape = cross_entropy_loss(out_shape, np.array([y_shape_true]))
            color_loss, grad_color = cross_entropy_loss(out_color, np.array([y_color_true]))
            loss += shape_loss + color_loss

            grad_fc_shape = self.fc_shape.backward(grad_shape, learning_rate)
            grad_relu3 = self.relu3.backward(grad_fc_shape)
            grad_fc1 = self.fc1.backward(grad_relu3, learning_rate)

            # Обратный проход для сверточных слоев не реализован

        return loss / batch_size
