import os
import io
from flask import Flask, jsonify, request, render_template
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from invert import Invert
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

UPLOAD_FOLDER = './static'

# Create app server
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define Net Class
class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 9, 3)
        self.fc1 = nn.Linear(9*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_size(x.size()))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    
    def num_size(self, size):
        num = size[1]*size[2]*size[3]
        return num

# Load saved model
model = NetCNN()
model.load_state_dict(torch.load('./my_model.pt'))
model.eval()

# Define transform image function
def transform_image(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize((28, 28)),
										transforms.Grayscale(1),
										Invert(),
										transforms.ToTensor(),
										transforms.Normalize((0.5,), (0.5,))
										])
	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)

# Define prediction function
def get_prediction(image_bytes, filename):
	img_tensor = transform_image(image_bytes=image_bytes)
	with torch.no_grad():
		logps = model(img_tensor)
	ps = torch.exp(logps)
	probab = list(ps.numpy()[0])
	plt.imshow(img_tensor[0].numpy().squeeze(), cmap='gray_r');
	plt.savefig('./static/' + filename)
	return probab.index(max(probab))

# Define route
@app.route("/", methods=['GET', 'POST'])
def predict():
	if request.method == 'GET':
		return render_template('index.html')
	elif request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			return render_template('index.html', data='no file apart')
		file = request.files['file']
		if file.filename == '':
			return render_template('index.html', data='no selected file')
		else:
			img_bytes = file.read()
			filename = file.filename
			return render_template('index.html', data=get_prediction(img_bytes, filename), img_url='/static/'+filename)

# Run Server
if __name__ == '__main__':
	app.run()


