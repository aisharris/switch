import cv2
import os
import shutil

def video_to_images(video_path=None, output_folder='frames', zip_name='frames.zip'):

    # Prompt for path to video
    # video_path = input("Enter path to video file: ")

    # output_folder = '{}_frames'.format(video_path.split('.')[0])
    # zip_name = '{}.zip'.format(output_folder)

    base_filename = os.path.basename(video_path).rsplit('.', 1)[0]

    # Create dir if doesnt exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Open video file and begin going through frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Break if at end of file
            break

        # Create and save frame
        frame_filename = os.path.join(output_folder, f'{base_filename}_frame_{count}.jpg')
        cv2.imwrite(frame_filename, frame)
        count += 1

    cap.release()

    # Create folder for zip dir
    zip_dir = os.path.dirname(zip_name) if os.path.dirname(zip_name) else '.'
    os.makedirs(zip_dir, exist_ok=True)

    # Make zip file
    base_name = zip_name.replace('.zip', '')
    shutil.make_archive(base_name, 'zip', '.', output_folder)

    print(f"Complete. Frames saved to {output_folder} and zipped to {zip_name}")

    # Remove extracted frames for optimal storage
    shutil.rmtree(output_folder)

    return

if __name__ == '__main__':
    video_to_images()