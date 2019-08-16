import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from labels import labels

class FashionNetPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, path):
        image = self.flatten_image(path)

        prediction = self.predict_(image)
        self.plot(prediction, image)

    def flatten_image(self, path):
        img = Image.open(path).convert('F')
        WIDTH, HEIGHT = img.size

        if WIDTH != 28 or HEIGHT != 28:
            img = img.resize((28, 28))
            
        WIDTH, HEIGHT = img.size

        print(img.size)

        data = list(img.getdata())
        data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = int(255 - data[i][j])
        
        return data

    def predict_(self, image):
        return self.model.predict([[image]])

    def plot_image_predict(self, prediction_vectors, image):
        prediction_vector = prediction_vectors[0]
        
        plt.grid(False)
        
        plt.xticks([])
        plt.yticks([])
        
        plt.imshow(image, cmap=plt.cm.binary)
        
        predicted_label = np.argmax(prediction_vector)
            
        plt.xlabel(
            'Guess: {} ({:2.0f} % certain)'.format(
                labels[predicted_label],
                100 * np.max(prediction_vector)
            )
        )

    def plot_value_array_predict(self, prediction_vectors):
        prediction_vector = prediction_vectors[0]
        
        plt.grid(False)
        
        plot = plt.bar(
            range(len(labels)),
            prediction_vector,
            color = '#fa34ab'
        )
        
        plt.xlabel('Clothing type')
        plt.ylabel('Certainty')
        
        plt.xticks(
            range(len(labels)),
            labels,
            size='small',
            rotation=90
        )
        
        plt.ylim([0, 1])

        nums = []
        for num in np.arange(0, 120, 20):
            nums.append(str(num) + "%")

        plt.yticks(np.arange(0, 1.2, .2), nums, size='small')

        predicted_label = np.argmax(prediction_vector)
        
        plot[predicted_label].set_color('red')
        
    def plot(self, prediction, image):
        plt.figure(figsize = (8, 4))
        plt.subplot(1, 2, 1)

        self.plot_image_predict(prediction, image)
        plt.subplot(1, 2, 2)

        self.plot_value_array_predict(prediction)

        plt.tight_layout()
        
        # plt.show()
        plt.savefig('somefile.png')

# net = NeuralNet()
# fashion_net = FashionNetPredictor(net, './data/custom_sneaker.jpg')