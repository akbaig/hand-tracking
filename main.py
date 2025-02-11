import cv2
import numpy as np
import mediapipe as mp
import random
import pygame

# ---------------------------------------------------
# Global Constants and Parameters
# ---------------------------------------------------
CANVAS_SIZE = (720, 1280, 3)  # (height, width, channels)
NUM_PARTICLES = 300           # Number of particles surrounding the control shape
MOTION_NOISE = 10             # Noise level in the prediction step
DEFAULT_CIRCLE_RADIUS = 30    # Default radius for the circle control shape
DEFAULT_SQUARE_SIDE = 60      # Default side length for the square control shape
DEFAULT_RECTANGLE_WIDTH = 80  # Default width for the rectangle control shape
DEFAULT_RECTANGLE_HEIGHT = 40 # Default height for the rectangle control shape
DEFAULT_TRIANGLE_SIDE = 60    # Default side length for triangle control shape

CANVAS_HEIGHT = CANVAS_SIZE[0]
CANVAS_WIDTH = CANVAS_SIZE[1]

# ---------------------------------------------------
# Game Object Classes
# ---------------------------------------------------

class Shape:
    def __init__(self):
        pass

    def draw(self, frame, color, thickness, filled):
        pass

    def get_radius(self):
        pass

    def contains(self, point, position):
        pass

    def get_position(self):
        return self.position
    
    def update_position(self, position):
        self.position = position

    @staticmethod
    def shape_name():
        pass

class Circle(Shape):
    def __init__(self, radius: int = DEFAULT_CIRCLE_RADIUS, position = None):
        self.radius = radius
        if position is None:
            # Randomly select x, y position within the canvas
            x = random.randint(radius, CANVAS_WIDTH - radius)
            y = random.randint(radius, CANVAS_HEIGHT - radius)
            position = (x, y)
        self.position = np.array(position)
        
    def draw(self, frame, color, thickness, filled):
        if filled:
            cv2.circle(frame, (int(self.position[0]), int(self.position[1])), self.radius, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (int(self.position[0]), int(self.position[1])), self.radius, color, thickness, cv2.LINE_AA)
    
    def get_radius(self):
        return self.radius
    
    def contains(self, point, position):
        return np.linalg.norm(point - position) < self.radius
    
    @staticmethod
    def shape_name():
        return "circle"

class Square(Shape):
    def __init__(self, length: int = DEFAULT_SQUARE_SIDE, position = None):
        self.side = length
        if position is None:
            # Randomly select x, y position within the canvas
            x = random.randint(length // 2, CANVAS_WIDTH - length // 2)
            y = random.randint(length // 2, CANVAS_HEIGHT - length // 2)
            position = (x, y)
        self.position = np.array(position)
    
    def update_position(self, position):
        self.position = position

    def draw(self, frame, color, thickness, filled):
        top_left = (int(self.position[0] - self.side // 2), int(self.position[1] - self.side // 2))
        bottom_right = (int(self.position[0] + self.side // 2), int(self.position[1] + self.side // 2))
        if filled:
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
        else:
            cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    
    def get_radius(self):
        return self.side // 2

    def contains(self, point, position):
        top_left = position - np.array([self.side / 2, self.side / 2])
        bottom_right = position + np.array([self.side / 2, self.side / 2])
        return (top_left[0] <= point[0] <= bottom_right[0]) and (top_left[1] <= point[1] <= bottom_right[1])
    
    @staticmethod
    def shape_name():
        return "square"

    
class Rectangle(Shape):
    def __init__(self, width: int = DEFAULT_RECTANGLE_WIDTH, height: int = DEFAULT_RECTANGLE_HEIGHT, position = None):
        self.width = width
        self.height = height
        if position is None:
            # Randomly select x, y position within the canvas
            x = random.randint(width // 2, CANVAS_WIDTH - width // 2)
            y = random.randint(height // 2, CANVAS_HEIGHT - height // 2)
            position = (x, y)
        self.position = np.array(position)

    def update_position(self, position):
        self.position = position

    def draw(self, frame, color, thickness, filled):
        top_left = (int(self.position[0] - self.width // 2), int(self.position[1] - self.height // 2))
        bottom_right = (int(self.position[0] + self.width // 2), int(self.position[1] + self.height // 2))
        if filled:
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
        else:
            cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    
    def get_radius(self):
        return max(self.width, self.height) // 2
    
    def contains(self, point, position):
        top_left = position - np.array([self.width / 2, self.height / 2])
        bottom_right = position + np.array([self.width / 2, self.height / 2])
        return (top_left[0] <= point[0] <= bottom_right[0]) and (top_left[1] <= point[1] <= bottom_right[1])
    
    @staticmethod
    def shape_name():
        return "rectangle"

class Triangle(Shape):
    def __init__(self, side: int = DEFAULT_TRIANGLE_SIDE, position = None):
        self.side = side
        if position is None:
            # Randomly select x, y position within the canvas
            x = random.randint(side, CANVAS_WIDTH - side)
            y = random.randint(side, CANVAS_HEIGHT - side)
            position = (x, y)
        self.position = np.array(position)
    
    def update_position(self, position):
        self.position = position

    def draw(self, frame, color, thickness, filled):
        pt1 = (int(self.position[0]), int(self.position[1] - int(self.side / np.sqrt(3))))
        pt2 = (int(self.position[0] - self.side // 2), int(self.position[1] + int(self.side / (2 * np.sqrt(3)))))
        pt3 = (int(self.position[0] + self.side // 2), int(self.position[1] + int(self.side / (2 * np.sqrt(3)))))
        pts = np.array([pt1, pt2, pt3], np.int32)
        if filled:
            cv2.fillPoly(frame, [pts], color)
        else:
            cv2.polylines(frame, [pts], True, color, thickness)
    
    def get_radius(self):
        return self.side
    
    def contains(self, point, position):
        pt1 = (position[0], position[1] - int(self.side / np.sqrt(3)))
        pt2 = (position[0] - self.side // 2, position[1] + int(self.side / (2 * np.sqrt(3))))
        pt3 = (position[0] + self.side // 2, position[1] + int(self.side / (2 * np.sqrt(3))))
        pts = np.array([pt1, pt2, pt3], np.int32)
        return cv2.pointPolygonTest(pts, (int(point[0]), int(point[1])), False) >= 0
    
    @staticmethod
    def shape_name():
        return "triangle"

class Particles:
    def __init__(self):
        # Select x, y position randomly within the canvas
        self.particles = np.random.randn(NUM_PARTICLES, 2) * 20
        self.particles += np.array([CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2])
    
    def resample_particles(self, weights):
        indices = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES, p=weights)
        return self.particles[indices]
    
    def update(self, measurement=None):
        # If a measurement (hand position) is provided, update the particles using importance sampling.
        if measurement is not None:
            distances = np.linalg.norm(self.particles - measurement, axis=1)
            weights = np.exp(-distances / 50)
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
                self.particles = self.resample_particles(weights)
        # Prediction step: random walk
        self.particles += np.random.randn(NUM_PARTICLES, 2) * MOTION_NOISE
        self.particles = np.clip(self.particles, [0, 0], [CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1])
    
    def draw(self, frame, color):
        for x, y in self.particles:
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)

class ParticleEffect:
    def __init__(self, position, color):
        self.particles = []
        self.color = color
        for _ in range(50):
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(2, 8)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            lifetime = random.randint(20, 40)
            self.particles.append({
                "pos": np.array(position, dtype=float),
                "vel": np.array([vx, vy], dtype=float),
                "lifetime": lifetime
            })

    def update_and_draw(self, frame):
        # Update each particle and draw it if still alive.
        alive_particles = []
        for particle in self.particles:
            particle["pos"] += particle["vel"]
            particle["lifetime"] -= 1
            if particle["lifetime"] > 0:
                pos = (int(particle["pos"][0]), int(particle["pos"][1]))
                cv2.circle(frame, pos, 2, self.color, -1)
                alive_particles.append(particle)
        self.particles = alive_particles
        return len(self.particles) > 0  # Return True if effect still active

class Obstacle:
    def __init__(self):
        w = random.randint(50, 100)
        h = random.randint(50, 100)
        x = random.randint(0, CANVAS_WIDTH - w)
        y = random.randint(0, CANVAS_HEIGHT - h)
        vx = random.choice([-5, -3, 3, 5])
        vy = random.choice([-5, -3, 3, 5])
        self.position = np.array([x, y], dtype=float)
        self.size = (w, h)
        self.velocity = np.array([vx, vy], dtype=float)

    def update(self):
        self.position += self.velocity
        if self.position[0] < 0 or self.position[0] + self.size[0] > CANVAS_WIDTH:
            self.velocity[0] = -self.velocity[0]
        if self.position[1] < 0 or self.position[1] + self.size[1] > CANVAS_HEIGHT:
            self.velocity[1] = -self.velocity[1]

    def draw(self, frame, color):
        top_left = (int(self.position[0]), int(self.position[1]))
        bottom_right = (int(self.position[0] + self.size[0]),
                        int(self.position[1] + self.size[1]))
        cv2.rectangle(frame, top_left, bottom_right, color, -1)

    def get_position(self):
        return self.position

# Shape names
SHAPES = [Circle.shape_name(), Square.shape_name(), Rectangle.shape_name(), Triangle.shape_name()]

# ---------------------------------------------------
# Main Game Class
# ---------------------------------------------------
class Game:
    def __init__(self):
        # Initialize Pygame Mixer for audio
        pygame.mixer.init()
        try:
            self.score_sound = pygame.mixer.Sound("score.wav")
            self.penalty_sound = pygame.mixer.Sound("penalty.mp3")
        except Exception as e:
            print("Could not load sound files:", e)
            self.score_sound = None
            self.penalty_sound = None

        # Initialize score, hearts, and bonus threshold tracker.
        self.score = 0
        self.hearts = 5
        self.last_heart_threshold = 0
        self.num_obstacles = 3

        # Colors
        self.goal_shape_color = (225, 105, 65) # Light Blue
        self.control_shape_color = self.goal_shape_color
        self.obstacle_color = (0, 0, 255) # Red
        self.particles_color = (100, 0, 0) # Blue
        self.particle_effect_color_success = self.goal_shape_color
        self.particle_effect_color_fail = self.obstacle_color

        # Initialize mediapipe and camera
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(min_detection_confidence=0.7,
                                                  min_tracking_confidence=0.7)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_HEIGHT)

        # Intialize particle filter
        self.particles = Particles()
        
        # Initialize game objects
        self.reset_round()

        # List to hold active particle effects
        self.effects = []

    def reset_round(self):
        """Reset the shape, holes, and obstacles for a new round."""
        self.control_shape = random.choice([Circle(), Square(), Rectangle(), Triangle()])
        self.goal_shapes = self.init_goal_shapes()
        self.obstacles = [Obstacle() for _ in range(self.num_obstacles)]
        self.collision_cooloff = 20 # Frames to wait before checking for collisions again

    def init_goal_shapes(self):
        """Initialize four shapes at the four corners with random offsets and unique shapes."""
        shapes = []
        four_corners = [(0, 0), (0, CANVAS_HEIGHT), (CANVAS_WIDTH, 0), (CANVAS_WIDTH, CANVAS_HEIGHT)]
        choices = [
            Circle(DEFAULT_CIRCLE_RADIUS + 20), 
            Square(DEFAULT_SQUARE_SIDE + 20), 
            Rectangle(DEFAULT_RECTANGLE_WIDTH + 20, DEFAULT_RECTANGLE_HEIGHT + 20), 
            Triangle(DEFAULT_TRIANGLE_SIDE + 20)
        ]
        for corner in four_corners:
            shape = random.choice(choices)
            # get shape radius as offset
            offset = shape.get_radius() + 10
            # Adjust offset based on corner's x-coordinate
            if corner[0] < CANVAS_WIDTH // 2:  # Left side
                offset_x = offset  # Ensure positive offset or smaller negative
            else:  # Right side
                offset_x = offset * -1  # Ensure negative offset or smaller positive
            # Adjust offset based on corner's y-coordinate
            if corner[1] < CANVAS_HEIGHT // 2:  # Top side
                offset_y = offset  # Ensure positive offset or smaller negative
            else:  # Bottom side
                offset_y = offset * -1  # Ensure negative offset or smaller positive
            new_position = (corner[0] + offset_x, corner[1] + offset_y)
            shape.update_position(np.array(new_position))
            shapes.append(shape)
            # remove the selected shape from the list
            choices.remove(shape)
        return shapes 

    def process_hand(self, frame):
        """Detect hand and return a measurement point if available."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)
        if results.multi_hand_landmarks:
            # Use first detected hand and index finger tip (landmark 9)
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[9]
            meas_x = int(index_tip.x * CANVAS_WIDTH)
            meas_y = int(index_tip.y * CANVAS_HEIGHT)
            return np.array([meas_x, meas_y])
        return None

    def check_goal_collisions(self):
        """Check collisions with goal shapes"""
        control_shape_position = self.control_shape.get_position()
        for goal_shape in self.goal_shapes:
            if goal_shape.contains(control_shape_position, goal_shape.get_position()):
                if self.control_shape.shape_name() == goal_shape.shape_name():
                    self.score += 1
                    if self.score_sound:
                        pygame.mixer.Sound.play(self.score_sound)
                    self.effects.append(ParticleEffect(goal_shape.get_position(), (0, 255, 0)))
                else:
                    self.hearts -= 1
                    if self.penalty_sound:
                        pygame.mixer.Sound.play(self.penalty_sound)
                    self.effects.append(ParticleEffect(goal_shape.get_position(), (0, 0, 255)))
                return True
        return False

    def check_obstacle_collisions(self):
        """Check collisions with obstacles, update hearts."""
        shape_position = self.control_shape.get_position()
        for obstacle in self.obstacles:
            obs_pos = obstacle.get_position()
            w, h = obstacle.size
            if (obs_pos[0] <= shape_position[0] <= obs_pos[0] + w) and (obs_pos[1] <= shape_position[1] <= obs_pos[1] + h):
                return True
        return False

    def check_collisions(self):
        """Check collisions with obstacles and goals, update score and hearts."""
        event_triggered = False

        # Check collision with obstacles
        if self.check_obstacle_collisions():
            event_triggered = True
            self.hearts -= 1
            self.effects.append(ParticleEffect(self.control_shape.get_position(), self.particle_effect_color_fail))
            if self.penalty_sound:
                pygame.mixer.Sound.play(self.penalty_sound)
        
        # Check collision with goals
        elif self.check_goal_collisions():
            # Award extra heart every 3 points
            event_triggered = True
            if (self.score // 3 > self.last_heart_threshold):
                self.hearts += 1
                self.last_heart_threshold = self.score // 3

        if event_triggered:
            self.reset_round()

        return event_triggered

    def control_shape_collides_obstacle(self, obstacle):
        """Check if the control shape's center is inside the obstacle rectangle."""
        pos = self.control_shape.get_position()
        obs_pos = obstacle.position
        w, h = obstacle.size
        return (obs_pos[0] <= pos[0] <= obs_pos[0] + w) and (obs_pos[1] <= pos[1] <= obs_pos[1] + h)

    def update_and_draw_effects(self, frame):
        """Update and draw active particle effects."""
        self.effects = [effect for effect in self.effects if effect.update_and_draw(frame)]

    def draw_obstacles(self, frame):
        for obs in self.obstacles:
            obs.draw(frame, self.obstacle_color)

    def run(self):

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # Mirror effect

            # Process hand detection and update control shape with measurement if available.
            measurement = self.process_hand(frame)
            self.particles.update(measurement)
            if measurement is not None:
                self.control_shape.update_position(measurement)

            # Update obstacles
            for obs in self.obstacles:
                obs.update()

            # Check for collisions (scoring or penalty events)
            if self.collision_cooloff > 0:
                self.collision_cooloff -= 1
            else:
                self.check_collisions()

            # Draw game elements
            for goal_shape in self.goal_shapes:
                goal_shape.draw(frame, self.goal_shape_color, 2, filled=False)
            self.control_shape.draw(frame, self.control_shape_color, 2, filled=True)
            self.draw_obstacles(frame)
            self.particles.draw(frame, self.particles_color)
            self.update_and_draw_effects(frame)

            # Display score and hearts
            cv2.putText(frame, f"Score: {self.score}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.goal_shape_color, 2)
            cv2.putText(frame, f"Hearts: {self.hearts}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.particle_effect_color_fail, 2)

            # Check for Game Over
            if self.hearts <= 0:
                cv2.putText(frame, "GAME OVER", (CANVAS_WIDTH // 2 - 150, CANVAS_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, self.particle_effect_color_fail, 4)
                cv2.imshow("Shape Matching", frame)
                cv2.waitKey(3000)
                break

            cv2.imshow("Shape Matching", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

# ---------------------------------------------------
# Main Execution
# ---------------------------------------------------
if __name__ == "__main__":
    game = Game()
    game.run()
