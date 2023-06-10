from flask import Flask, render_template, request, url_for
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath('static/uploads')

def process_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_video.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    while success:
        # Your frame processing code
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        lower_red = np.array([0, 31, 255])
        upper_red = np.array([176, 255, 255])
        lower_white = np.array([0, 0, 0])
        upper_white = np.array([0, 0, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        res = cv2.bitwise_and(image, image, mask=mask)
        res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((13, 13), np.uint8)
        thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h >= (1.5 * w):
                if w > 15 and h >= 15:
                    player_img = image[y:y + h, x:x + w]
                    player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                    mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                    res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                    nzCount = cv2.countNonZero(res1)
                    mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                    res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
                    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
                    nzCountred = cv2.countNonZero(res2)
                    if nzCount >= 20:
                        cv2.putText(image, 'France', (x - 2, y - 2), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    else:
                        pass
                    if nzCountred >= 20:
                        cv2.putText(image, 'Belgium', (x - 2, y - 2), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    else:
                        pass

        # Write the frame into the output video
        out.write(image)
        success, image = vidcap.read()

    # Release the video objects
    vidcap.release()
    out.release()
    cv2.destroyAllWindows()

    # Return the path to the output video and team names
    return output_video_path, 'France', 'Belgium'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files['video']
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)

        output_video_path, team1_name, team2_name = process_video(video_path)
        # Extract only the file name
        output_video_filename = os.path.basename(output_video_path)

        return render_template('index.html', video_url=url_for('static', filename=f'uploads/{output_video_filename}'),
                               team1_name=team1_name, team2_name=team2_name)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
