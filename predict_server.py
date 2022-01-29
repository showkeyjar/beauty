# %% coding=utf-8
import logging
from PIL import Image
from os.path import basename
from flask import Flask
from flask import render_template, request
from werkzeug.utils import secure_filename
from face_report import predict, gen_report_file
from utils import *
from feature.tools import face_correct

app = Flask(__name__)
uploads_dir = "static/uploads"

"""
颜值评测
"""
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d: %(message)s')
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return render_template('index.htm')


@app.route('/list')
def report_list():
    """
    报告列表
    :return:
    """
    # 检查是否有报告
    reports = get_files('templates/reports/', '.htm')
    return render_template('list.htm', reports=reports)


@app.route('/view')
def report_view():
    """
    预览报告
    :return:
    """
    filename = request.args.get('id')
    return render_template('reports/' + filename)


@app.route('/plan')
def plan():
    """
    改进方案
    :return:
    """
    return render_template('plan.htm')


@app.route('/pk')
def face_pk():
    """
    颜值pk
    :return:
    """
    return render_template('pk.htm')


@app.route('/upload_gen', methods=['GET', 'POST'])
def upload_gen():
    """
    通过shap解释服务
    :return:
    """
    profile = request.files['sc']
    save_file = os.path.join(uploads_dir, secure_filename(profile.filename))
    profile.save(save_file)
    gen_report_file(save_file, secure_filename(profile.filename))
    return render_template('reports/' + secure_filename(profile.filename) + '.htm')


@app.route('/gen', methods=['GET', 'POST'])
def gen():
    if request.method == 'POST':
        im_path = request.form['im_path']
        logger.debug(im_path)
        file_name = secure_filename(basename(im_path))
        type = request.form['type']
        gen_report_file(im_path, file_name, type)
        return render_template('reports/' + file_name + '.htm')
    return render_template('t1.htm')


@app.route('/pred', methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':
        # save the single "profile" file
        profile = request.files['sc']
        save_file = os.path.join(uploads_dir, secure_filename(profile.filename))
        #矫正图片
        profile.save(save_file)
        new_path = face_correct(save_file)
        # img.save(save_file)
        pred_score = predict(new_path)
        return render_template('t1.htm', score=pred_score, img=new_path)
        #return redirect(url_for('upload'))
    return render_template('t1.htm')


if __name__ == "__main__":
    app.run(debug=True)
