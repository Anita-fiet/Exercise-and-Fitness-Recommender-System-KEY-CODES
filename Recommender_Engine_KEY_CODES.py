# Import the necessary libraries
import flask
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define the hyperparameters
learning_rate = 0.01
epochs = 100
batch_size = 32
hidden_size = 256
num_classes = 9

# Load the data from the database
df = pd.read_csv('megaGymDataset.csv')
X = df.drop('Exercise', axis=1).values
y = df['Exercise'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(hidden_size // 2, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Save the model
model.save('fitness_model.h5')

# Create a Flask app
app = flask.Flask(__name__)


# Define a route for the home page
@app.route('/')
def home():
    return flask.render_template('home.html')


# Define a route for the recommendation page
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the user input from the form
    user_input = flask.request.form.to_dict()

    # Convert the user input to a numpy array
    user_input = np.array(list(user_input.values())).reshape(1, -1)

    # Load the model
    model = tf.keras.models.load_model('fitness_model.h5')

    # Make a prediction
    prediction = model.predict(user_input)

    # Get the index of the highest probability class
    prediction = np.argmax(prediction)

    # Map the index to the class name
    classes = ['Yoga', 'Running', 'Cycling', 'Swimming', 'Weightlifting', 'HIIT', 'Zumba', 'Boxing', 'Plank']
    recommendation = classes[prediction]

    # Return the recommendation to the user
    return flask.render_template('recommend.html', recommendation=recommendation)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
