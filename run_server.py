from app import app, load_model

# Load model first
load_model()

# Run with threaded=True to handle multiple requests
app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
