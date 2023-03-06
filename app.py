from flask import Flask, request, jsonify
import tensorflow as tf
# from flask_restful import Resource, Api, reqparse
import werkzeug
import numpy as np
import cv2

app = Flask(__name__)
     

# Load the TFLite model
# ----
path = "mobilenetv2.tflite"
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()

input_tensor = interpreter.get_input_details()[0]["index"]
output_tensor = interpreter.get_output_details()[0]["index"]

class_names = ['Grassy Shoots', 'Healthy', 'Mites', 'Ring Spot', 'YLD']

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for predicting the class and probability of an image using a TFLite model.
    ---
    parameters:
        - in : formData
            name: image
            type: file
            required: true
    """

    # Get the image file from the request
    image = request.files["image"]
    
    # Save the image to a specified path
    image_path = "images/saved_image.jpeg"
    image.save(image_path)
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img , 0)
    
    # Get the predictions from the TFLite model
    interpreter.set_tensor(input_tensor, img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_tensor)[0]
    
    # Get the class index and probability with the highest predicted probability
    class_index = np.argmax(predictions)
    probability = predictions[class_index]
    
    # Return the class index and probability as a JSON object
    return jsonify(class_index=str(class_index), probability=str(probability))

if __name__ == "__main__":
    app.run(debug=True)
