import os
import pickle
import numpy as np
from nn.cnn import CNN
from data.generated_data import generate_dataset  # функция генерации датасета, должна быть в другом модуле
import time

MODEL_PATH = "data/saved_model.pkl"

class NNController:
    def __init__(self):
        self.model = CNN()
        if os.path.exists(MODEL_PATH):
            self.load_model()

    def train(self, epochs=5, batch_size=16, lr=0.001):
        for epoch in range(epochs):
            X, y_shape, y_color = generate_dataset(num_samples=1000, img_size=64)
            losses = []
            start_time = time.time()
            num_batches = len(X) // batch_size
            print(f"Epoch {epoch+1}/{epochs}")

            for i in range(0, len(X), batch_size):
                x_batch = X[i:i+batch_size]
                y_shape_batch = y_shape[i:i+batch_size]
                y_color_batch = y_color[i:i+batch_size]

                loss = self.model.train_batch(x_batch, y_shape_batch, y_color_batch, learning_rate=lr)
                losses.append(loss)

                # Лог прогресса каждые 10 батчей или в конце эпохи
                if (i // batch_size) % 10 == 0 or i + batch_size >= len(X):
                    avg_loss = np.mean(losses)
                    print(f"  Batch {(i // batch_size) + 1}/{num_batches}, Avg Loss: {avg_loss:.4f}")

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s, Avg Loss: {np.mean(losses):.4f}\n")
            self.save_model()


    def predict(self, image):
        import cv2
        img_resized = cv2.resize(image, (64,64))
        return self.model.predict(img_resized)

    def save_model(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
