import os
import cv2
import numpy as np
import face_recognition
import yaml
import pickle
import time

class FaceRecognizer:
    def __init__(self, config_path='config/settings.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.face_config = self.config['face_recognition']
        self.face_data_path = self.config['paths']['face_data']
        
        # Ensure the face data directory exists
        os.makedirs(self.face_data_path, exist_ok=True)
        
        # Parameters
        self.tolerance = self.face_config['tolerance']
        self.camera_index = self.face_config['camera_index']
        self.frame_width = self.face_config['frame_width']
        self.frame_height = self.face_config['frame_height']
        self.scale_factor = self.face_config['scale_factor']
        
        # Initialize camera
        self.camera = None
        
        # Known face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load saved face encodings if available
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load known face encodings from the data directory."""
        encodings_file = os.path.join(self.face_data_path, 'encodings.pkl')
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'rb') as file:
                    data = pickle.load(file)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_encodings)} face encodings")
            except Exception as e:
                print(f"Error loading face encodings: {e}")
    
    def _save_known_faces(self):
        """Save known face encodings to the data directory."""
        encodings_file = os.path.join(self.face_data_path, 'encodings.pkl')
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(encodings_file, 'wb') as file:
            pickle.dump(data, file)
    
    def _initialize_camera(self):
        """Initialize the camera if not already initialized."""
        if self.camera is None:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # Check if camera opened successfully
            if not self.camera.isOpened():
                print(f"Error: Could not open camera at index {self.camera_index}")
                print("Try running 'python src/test_camera.py --scan' to find available cameras")
                return False
                
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            return True
        return True
    
    def _release_camera(self):
        """Release the camera if initialized."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def test_camera(self, display_time=10):
        """
        Test if the camera is working properly.
        
        Args:
            display_time (int): How long to display the video feed in seconds
            
        Returns:
            bool: True if camera works, False otherwise
        """
        print(f"Testing camera at index {self.camera_index}...")
        
        if not self._initialize_camera():
            return False
        
        # Get camera properties
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera properties: {width}x{height} at {fps} FPS")
        print(f"Showing camera feed for {display_time} seconds. Press 'q' to quit earlier.")
        
        start_time = time.time()
        frame_count = 0
        success = False
        
        while time.time() - start_time < display_time:
            # Capture frame-by-frame
            ret, frame = self.camera.read()
            
            # If frame is read correctly, ret is True
            if not ret:
                print("Error: Couldn't read frame.")
                time.sleep(0.1)  # Small delay to avoid busy-waiting
                continue
            
            success = True  # We got at least one frame
            frame_count += 1
            
            # Display info on the frame
            cv2.putText(frame, f"Camera {self.camera_index}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Attempt to detect faces
            small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # Draw rectangles around detected faces
            for top, right, bottom, left in face_locations:
                # Scale back up face locations
                top = int(top / self.scale_factor)
                right = int(right / self.scale_factor)
                bottom = int(bottom / self.scale_factor)
                left = int(left / self.scale_factor)
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Display face detection status
            if face_locations:
                cv2.putText(frame, f"Detected {len(face_locations)} face(s)", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No faces detected", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Face Recognition Test', frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Calculate actual FPS
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"Actual FPS: {actual_fps:.2f}")
        print(f"Face detection {'working' if success else 'not working'}")
        
        # Release the camera and close the window
        cv2.destroyAllWindows()
        
        # Don't release the camera here as it might be needed later
        
        return success
    
    def enroll_user(self, user_name, num_samples=5, delay=1, confirm=True):
        """
        Enroll a new user by capturing face samples.
        
        Args:
            user_name (str): Name of the user to enroll
            num_samples (int): Number of face samples to capture
            delay (int): Delay between captures in seconds
            confirm (bool): Whether to ask for confirmation before saving
        
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        if not self._initialize_camera():
            return False
        
        # Create directory for debug images
        debug_dir = os.path.join(self.face_data_path, f"debug_{user_name}_{int(time.time())}")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Saving debug images to: {debug_dir}")
        
        # Enrollment loop - continue until user is satisfied or cancels
        enrollment_complete = False
        
        while not enrollment_complete:
            face_encodings = []
            captured_frames = []
            
            print(f"\nEnrolling user: {user_name}")
            print(f"Capturing {num_samples} face samples. Please look at the camera.")
            
            # First check if camera is working and not frozen
            print("Testing camera stream before enrollment...")
            prev_frame = None
            identical_frames = 0
            max_identical = 5  # Maximum number of identical frames before warning
            
            # Capture a few frames to warm up the camera
            for _ in range(10):
                ret, frame = self.camera.read()
                time.sleep(0.1)
            
            # Check for identical frames (possible frozen camera)
            for i in range(10):
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture image from camera during test")
                    continue
                    
                # Save first test frame
                if i == 0:
                    test_frame_path = os.path.join(debug_dir, f"camera_test.jpg")
                    cv2.imwrite(test_frame_path, frame)
                    
                # Compare with previous frame if available
                if prev_frame is not None:
                    # Calculate difference between frames
                    diff = cv2.absdiff(frame, prev_frame)
                    non_zero_count = np.count_nonzero(diff)
                    total_pixels = frame.shape[0] * frame.shape[1] * frame.shape[2]
                    diff_percentage = (non_zero_count / total_pixels) * 100
                    
                    print(f"Frame {i} difference: {diff_percentage:.2f}%")
                    
                    # If frames are very similar, increment counter
                    if diff_percentage < 0.1:  # Less than 0.1% difference
                        identical_frames += 1
                        if identical_frames >= max_identical:
                            print("WARNING: Camera may be frozen - detecting nearly identical frames")
                            # Save the problematic frames
                            cv2.imwrite(os.path.join(debug_dir, f"frozen_test_1.jpg"), prev_frame)
                            cv2.imwrite(os.path.join(debug_dir, f"frozen_test_2.jpg"), frame)
                
                prev_frame = frame.copy()
                time.sleep(0.1)
            
            # Proceed with enrollment
            for i in range(num_samples):
                print(f"Capturing sample {i+1}/{num_samples} in 3 seconds...")
                
                # Countdown with visible feedback
                for j in range(3, 0, -1):
                    print(f"{j}...")
                    ret, countdown_frame = self.camera.read()
                    if ret:
                        cv2.putText(countdown_frame, f"Capturing in {j}...", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Enrollment', countdown_frame)
                        cv2.waitKey(1)
                    time.sleep(1)
                    
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture image from camera")
                    # Save empty frame for debugging
                    empty_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    cv2.putText(empty_frame, "CAPTURE FAILED", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(debug_dir, f"failed_capture_{i+1}.jpg"), empty_frame)
                    continue
                
                # Save the raw captured frame (regardless of face detection)
                raw_frame_path = os.path.join(debug_dir, f"raw_frame_{i+1}.jpg")
                cv2.imwrite(raw_frame_path, frame)
                print(f"Saved raw frame {i+1} to {raw_frame_path}")
                
                # Check if this frame is identical to previous frames
                if captured_frames:
                    for j, prev_frame in enumerate(captured_frames):
                        diff = cv2.absdiff(frame, prev_frame)
                        non_zero_count = np.count_nonzero(diff)
                        total_pixels = frame.shape[0] * frame.shape[1] * frame.shape[2]
                        diff_percentage = (non_zero_count / total_pixels) * 100
                        print(f"Frame {i+1} difference from frame {j+1}: {diff_percentage:.2f}%")
                        
                        if diff_percentage < 0.5:  # Less than 0.5% difference
                            print(f"WARNING: Frame {i+1} is very similar to frame {j+1}")
                
                # Store this frame for later comparison
                captured_frames.append(frame.copy())
                
                # Convert BGR to RGB (face_recognition uses RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find face locations in the frame
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if not face_locations:
                    print("No face detected. Please ensure your face is visible to the camera.")
                    
                    # Save the no-face frame with annotation
                    no_face_frame = frame.copy()
                    cv2.putText(no_face_frame, "No face detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(debug_dir, f"no_face_{i+1}.jpg"), no_face_frame)
                    
                    # Show the frame so user can see what the camera sees
                    cv2.imshow('Enrollment', no_face_frame)
                    cv2.waitKey(1000)  # Wait 1 second
                    continue
                
                # If multiple faces are detected, use the largest face (closest to camera)
                if len(face_locations) > 1:
                    print(f"Multiple faces detected. Using the largest face.")
                    # Calculate area of each face and find the largest
                    areas = [(right-left)*(bottom-top) for top, right, bottom, left in face_locations]
                    largest_face_idx = areas.index(max(areas))
                    face_locations = [face_locations[largest_face_idx]]
                
                # Get face encodings
                current_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                if current_encodings:
                    face_encodings.append(current_encodings[0])
                    
                    # Extract face region for saving
                    top, right, bottom, left = face_locations[0]
                    face_img = frame[top:bottom, left:right]
                    face_img_path = os.path.join(debug_dir, f"face_{i+1}.jpg")
                    cv2.imwrite(face_img_path, face_img)
                    print(f"Saved face {i+1} to {face_img_path}")
                    
                    # Draw rectangle on full frame and save
                    annotated_frame = frame.copy()
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Sample {i+1} captured", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(debug_dir, f"detected_face_{i+1}.jpg"), annotated_frame)
                    
                    # Display the annotated frame
                    cv2.imshow('Enrollment', annotated_frame)
                    cv2.waitKey(1000)  # Show the successful capture for 1 second
                    
                    print(f"Sample {i+1} captured successfully")
                else:
                    print(f"Failed to encode face in sample {i+1}")
                    # Save failed encoding frame
                    failed_frame = frame.copy()
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(failed_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(failed_frame, "Failed to encode", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(debug_dir, f"failed_encoding_{i+1}.jpg"), failed_frame)
            
            cv2.destroyAllWindows()
            
            # Check if we have enough face samples
            if len(face_encodings) == 0:
                print(f"Failed to enroll user {user_name}. No valid face encodings captured.")
                
                if confirm:
                    retry = input("\nNo faces were detected. Would you like to try again? (y/n): ").lower().strip()
                    if retry == 'y' or retry == 'yes':
                        continue
                    else:
                        return False
                else:
                    return False
            
            # If not enough samples were captured, ask if the user wants to proceed
            if len(face_encodings) < num_samples and confirm:
                print(f"\nOnly {len(face_encodings)}/{num_samples} valid face samples were captured.")
                proceed = input("Would you like to proceed with these samples or try again? (p/t): ").lower().strip()
                
                if proceed == 't' or proceed == 'try':
                    continue  # Try enrollment again
            
            # Ask for confirmation if requested
            if confirm:
                # Show a summary of the captured faces
                summary_image = None
                
                # Create a grid of captured faces if available
                if len(face_encodings) > 0:
                    # Collect all successful face images
                    faces = []
                    for i in range(1, num_samples + 1):
                        face_path = os.path.join(debug_dir, f"face_{i}.jpg")
                        if os.path.exists(face_path):
                            face = cv2.imread(face_path)
                            if face is not None:
                                # Resize to common size
                                face = cv2.resize(face, (150, 150))
                                faces.append(face)
                    
                    if faces:
                        # Create a grid layout
                        rows = (len(faces) + 2) // 3  # Ceiling division
                        cols = min(3, len(faces))
                        
                        # Create blank canvas
                        summary_image = np.zeros((rows * 150, cols * 150, 3), dtype=np.uint8)
                        
                        # Place faces in grid
                        for i, face in enumerate(faces):
                            r, c = i // cols, i % cols
                            summary_image[r*150:(r+1)*150, c*150:(c+1)*150] = face
                        
                        # Show the summary image
                        cv2.imshow('Enrollment Summary', summary_image)
                        cv2.waitKey(1)
                
                # Ask user to confirm enrollment
                confirmation = input(f"\nDo you want to save this enrollment for {user_name}? (y/n): ").lower().strip()
                
                if summary_image is not None:
                    cv2.destroyAllWindows()
                
                if confirmation != 'y' and confirmation != 'yes':
                    retry = input("Would you like to try again? (y/n): ").lower().strip()
                    if retry == 'y' or retry == 'yes':
                        continue
                    else:
                        return False
            
            # User confirmed or confirmation not required
            if face_encodings:
                # Calculate the average encoding for more robust recognition
                average_encoding = np.mean(face_encodings, axis=0)
                
                # Add to known faces
                self.known_face_encodings.append(average_encoding)
                self.known_face_names.append(user_name)
                
                # Save the updated encodings
                self._save_known_faces()
                
                print(f"User {user_name} enrolled successfully with {len(face_encodings)} face samples")
                print(f"Debug images saved to {debug_dir}")
                
                # Set flag to exit the loop
                enrollment_complete = True
                return True
            else:
                print(f"Failed to enroll user {user_name}. No valid face encodings captured.")
                return False
    
    def recognize_face(self, display_video=False, recognition_time=5):
        """
        Recognize a face from the camera feed.
        
        Args:
            display_video (bool): Whether to display the video feed
            recognition_time (int): Time in seconds to attempt recognition
        
        Returns:
            tuple: (recognized, user_name) - whether a face was recognized and the user's name
        """
        if not self.known_face_encodings:
            print("No known faces. Please enroll a user first.")
            return False, None
        
        if not self._initialize_camera():
            return False, None
        
        start_time = time.time()
        best_match = None
        best_match_name = None
        
        while time.time() - start_time < recognition_time:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture image from camera")
                time.sleep(0.1)  # Small delay to avoid busy-waiting
                continue
            
            # Resize frame for faster face detection
            small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations in the frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if not face_locations:
                if display_video:
                    cv2.putText(frame, "No face detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            # If multiple faces are detected, use the largest face (closest to camera)
            if len(face_locations) > 1:
                # Calculate area of each face and find the largest
                areas = [(right-left)*(bottom-top) for top, right, bottom, left in face_locations]
                largest_face_idx = areas.index(max(areas))
                face_locations = [face_locations[largest_face_idx]]
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            if face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[0], 
                                                         tolerance=self.tolerance)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encodings[0])
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    name = self.known_face_names[best_match_index]
                    
                    # Keep track of the best match overall
                    if best_match is None or face_distances[best_match_index] < best_match:
                        best_match = face_distances[best_match_index]
                        best_match_name = name
                    
                    if display_video:
                        # Draw a rectangle around the face
                        top, right, bottom, left = face_locations[0]
                        # Scale back up face locations
                        top = int(top / self.scale_factor)
                        right = int(right / self.scale_factor)
                        bottom = int(bottom / self.scale_factor)
                        left = int(left / self.scale_factor)
                        
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    if display_video:
                        top, right, bottom, left = face_locations[0]
                        # Scale back up face locations
                        top = int(top / self.scale_factor)
                        right = int(right / self.scale_factor)
                        bottom = int(bottom / self.scale_factor)
                        left = int(left / self.scale_factor)
                        
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if display_video:
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if display_video:
            cv2.destroyAllWindows()
        
        self._release_camera()
        
        if best_match_name:
            return True, best_match_name
        else:
            return False, None


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Recognition Module')
    parser.add_argument('--test-camera', action='store_true',
                        help='Test camera and face detection')
    parser.add_argument('--display-time', type=int, default=10,
                        help='Time in seconds to display camera feed during test')
    parser.add_argument('--enroll', type=str, metavar='NAME',
                        help='Enroll a new user with the given name')
    parser.add_argument('--recognize', action='store_true',
                        help='Start face recognition')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of face samples to capture during enrollment')
    
    args = parser.parse_args()
    
    # Create face recognizer
    recognizer = FaceRecognizer()
    
    if args.test_camera:
        # Test the camera
        recognizer.test_camera(display_time=args.display_time)
    
    elif args.enroll:
        # Enroll a new user
        recognizer.enroll_user(args.enroll, num_samples=args.samples)
    
    elif args.recognize:
        # Recognize faces
        print("Starting face recognition...")
        recognized, name = recognizer.recognize_face(display_video=True)
        
        if recognized:
            print(f"Recognized: {name}")
        else:
            print("No face recognized")
    
    else:
        # Default behavior - check if there are enrolled users, then test
        if not recognizer.known_face_names:
            print("No enrolled users found.")
            print("Try testing the camera first with: python src/face_recognition_module.py --test-camera")
            print("Then enroll a user with: python src/face_recognition_module.py --enroll \"Your Name\"")
        else:
            print(f"Found {len(recognizer.known_face_names)} enrolled user(s).")
            print("Start recognition with: python src/face_recognition_module.py --recognize")
        
        # Always test the camera by default
        recognizer.test_camera(display_time=5)
