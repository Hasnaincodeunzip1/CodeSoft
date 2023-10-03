import csv
import pickle
import face_recognition  # Assuming you're using this library for face recognition

# Load your CSV database
data = []  # List to store (name, embedding) pairs

with open('student.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        name = row['Name']
        image_path = row['']  # Assuming you have a column for image paths
        image = face_recognition.load_image_file(image_path)
        embeddings = face_recognition.face_encodings(image)
        if len(embeddings) > 0:
            embedding = embeddings[0]  # Assuming one face per image
            data.append((name, embedding))

# Save the new embeddings to a pickle file
with open('embeddings.pickle', 'wb') as f:
    pickle.dump(data, f)
