// Load the pre-trained model (MobileNet in this case)
let model;
(async function() {
  model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  console.log('Model loaded');
})();

// Handle image upload and display
const imageInput = document.getElementById('imageInput');
const uploadedImage = document.getElementById('uploadedImage');
imageInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.onload = () => {
    uploadedImage.src = reader.result;
    uploadedImage.style.display = 'block';
  };
  reader.readAsDataURL(file);
});

// Classify the image when the button is clicked
const classifyButton = document.getElementById('classifyButton');
const predictionResult = document.getElementById('predictionResult');
classifyButton.addEventListener('click', async () => {
  if (uploadedImage.src) {
    // Pre-process the image for the model
    const imgTensor = tf.browser.fromPixels(uploadedImage)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .expandDims();

    // Make prediction
    const predictions = await model.predict(imgTensor).data();
    const topPrediction = Array.from(predictions).indexOf(Math.max(...predictions));

    // Display result
    predictionResult.textContent = `Prediction: ${topPrediction}`;
  } else {
    alert('Please upload an image first.');
  }
});
