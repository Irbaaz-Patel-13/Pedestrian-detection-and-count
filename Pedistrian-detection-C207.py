import cv2
import imutils
import os

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Paths for input and output videos
input_video_path = '/Users/irbaazpatel/PycharmProjects/Pedistrian-tracking/2121-155244120_small.mp4'
output_video_path = '/Users/irbaazpatel/PycharmProjects/Pedistrian-tracking/pedestrian_detection_output_with_count.mp4'

# Check if the input video exists
if not os.path.exists(input_video_path):
    raise FileNotFoundError(f"Input video not found: {input_video_path}")

# Open the video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {input_video_path}")

# Retrieve video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object for saving
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for macOS-compatible .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Total pedestrian count
total_pedestrians = 0

# Process each frame in the video
frame_count = 0  # For tracking processed frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_count += 1
    # Resize the frame for faster processing (adjust width)
    resized_frame = imutils.resize(frame, width=min(600, frame.shape[1]))

    # Detect people in the frame
    (rects, weights) = hog.detectMultiScale(
        resized_frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # Update total pedestrian count
    total_pedestrians += len(rects)

    # Draw bounding boxes for detected pedestrians
    for (x, y, w, h) in rects:
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Optionally add a label to the rectangle
        cv2.putText(
            resized_frame,
            "Pedestrian",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Display total pedestrian count on the frame
    cv2.putText(
        resized_frame,
        f"Total Pedestrians: {total_pedestrians}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Write the processed frame to the output video
    processed_frame = cv2.resize(resized_frame, (width, height))  # Match original size
    out.write(processed_frame)

    # Optional: Display the output frame in real-time
    cv2.imshow('Pedestrian Detection', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Display total pedestrian count after processing
print(f"Processed {frame_count} frames.")
print(f"Total pedestrians detected in the video: {total_pedestrians}")
print(f"Processed video saved to: {output_video_path}")
