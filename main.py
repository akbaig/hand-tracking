import cv2
import numpy as np

# Number of particles
N_PARTICLES = 300

# Initialize particles randomly
def initialize_particles(frame):
    h, w, _ = frame.shape
    particles = np.c_[np.random.uniform(0, w, N_PARTICLES),
                      np.random.uniform(0, h, N_PARTICLES)]
    return particles

# Motion model (random movement)
def predict_particles(particles):
    noise = np.random.randn(N_PARTICLES, 2) * 5  # Gaussian noise
    particles += noise
    return particles

# Likelihood measurement using simple color detection
def compute_weights(particles, frame):
    weights = np.ones(N_PARTICLES)
    
    for i, (x, y) in enumerate(particles):
        x, y = int(x), int(y)
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            b, g, r = frame[y, x]
            # Example: Detect skin color (this can be improved with ML models)
            if 100 < r < 255 and 50 < g < 180 and 30 < b < 150:  
                weights[i] = 1.0  # Assign high weight for skin-color pixels
            else:
                weights[i] = 0.1  # Low weight for non-skin pixels

    weights += 1e-5  # Avoid zero weights
    weights /= np.sum(weights)  # Normalize
    return weights

# Resample particles based on weights
def resample_particles(particles, weights):
    indices = np.random.choice(range(N_PARTICLES), size=N_PARTICLES, p=weights)
    return particles[indices]

# Compute weighted mean for estimation
def estimate_position(particles, weights):
    return np.average(particles, weights=weights, axis=0)

# Main function to process video
cap = cv2.VideoCapture(0)
particles = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if particles is None:
        particles = initialize_particles(frame)

    # Predict particles
    particles = predict_particles(particles)

    # Compute weights
    weights = compute_weights(particles, frame)

    # Resample
    particles = resample_particles(particles, weights)

    # Estimate hand position
    estimated_position = estimate_position(particles, weights)

    # Draw particles
    for x, y in particles:
        cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), -1)

    # Draw estimated position
    cv2.circle(frame, (int(estimated_position[0]), int(estimated_position[1])), 10, (0, 255, 0), -1)

    # Display
    cv2.imshow("Particle Filter Hand Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
