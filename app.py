from flask import Flask, render_template, flash, request, redirect, url_for
import os
import cv2
from werkzeug.utils import secure_filename
from volumecalc import obtain_point_cloud, compute_volume, obtain_depth_data
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "upload")

app.config['SECRET_KEY'] = "johncuvier"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 2048 * 2048


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_img():
    """
    Description: Get user data from index.html, if the file is valid obtain further images that are needed to compute point_cloud. Then by using this point cloud calculates what is the volume and print to the terminal.
    """

    if 'original' not in request.files:
        flash("No file part")
        return redirect(request.url)

    # Get only the original image and print its id
    original_img = request.files['original']
    original_filename = secure_filename(original_img.filename)
    img_id = original_filename.split('_')[0]
    # When it is possible, do not use predefined paths for your project.
    folder_path = "Data"
    # It is assumed that later in project phase, best depth image can be obtained while processing.
    depth_img, mask, inpainted = obtain_depth_data(folder_path, img_id)
    point_cloud = obtain_point_cloud(depth_img, inpainted, mask)
    volume = compute_volume(point_cloud)

    # Save image to this path
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    original_img.save(save_path)
    # The results were not close to good, we discussed during last meeting
    M3TOCM3 = 1000000  # 1m^3 = 10^6cm3
    print(f"The volume under the given food is {round(volume * M3TOCM3)} cm3")
    return render_template('index.html', filename=original_filename, volume=round(volume * M3TOCM3))


@app.route("/display/<filename>")
def display_img(filename):
    print("Displayed image", filename)
    return redirect(url_for('static', filename='upload/' + filename), code=301)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

# What I need to do now

# An image is given from the interface
# Get the image and obtain its parameters etc.
