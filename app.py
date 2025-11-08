import os
import cv2
import torch
import glob
from flask import Flask, request, send_file, render_template, jsonify
from werkzeug.utils import secure_filename

try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- 初始化 Flask 应用 ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- 全局变量和模型缓存 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
upsampler_cache = {}
gfpgan_cache = None


def get_gfpganer():
    global gfpgan_cache
    if gfpgan_cache is None and GFPGANer is not None:
        model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'

        # #######################################################
        # ###############  核心逻辑修正 1 #########################
        # #######################################################
        # 将 upscale 设为 1，让 GFPGAN 只修复人脸，不进行额外的放大。
        gfpgan_cache = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=device)
    return gfpgan_cache


def get_upsampler(model_name):
    if model_name in upsampler_cache:
        return upsampler_cache[model_name]
    if 'RealESRGAN_x4plus_anime_6B' in model_name:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )
    upsampler_cache[model_name] = upsampler
    return upsampler


# --- 路由 ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/models', methods=['GET'])
def get_models():
    model_files = glob.glob('weights/*.pth')
    model_names = [os.path.splitext(os.path.basename(f))[0] for f in model_files]
    return jsonify(model_names)


@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'file' not in request.files: return 'No file part', 400
    file = request.files['file']
    if file.filename == '': return 'No selected file', 400

    model_name = request.form.get('model', 'RealESRGAN_x4plus')
    outscale = float(request.form.get('outscale', 4))
    face_enhance = request.form.get('face_enhance') == 'true'

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None: return 'Cannot read input image', 500

        try:
            upsampler = get_upsampler(model_name)

            # #######################################################
            # ###############  核心逻辑修正 2 #########################
            # #######################################################

            # 1. 始终先用 RealESRGAN 放大到模型原始的 4x，以获取最佳质量
            output, _ = upsampler.enhance(img, outscale=4)

            # 2. 如果需要，在 4x 的高质量结果上进行人脸增强
            if face_enhance and GFPGANer is not None:
                gfpgan = get_gfpganer()
                _, _, output = gfpgan.enhance(output, has_aligned=False, only_center_face=False, paste_back=True)

            # 3. 最后，根据用户选择的 outscale，对结果进行精确的缩放
            if outscale != 4:
                # 计算目标尺寸
                target_height = int(img.shape[0] * outscale)
                target_width = int(img.shape[1] * outscale)
                # 使用高质量的 Lanczos 插值算法进行缩放
                output = cv2.resize(output, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

        except FileNotFoundError as e:
            print(e);
            return str(e), 500
        except Exception as e:
            print(f"An error occurred: {e}");
            return 'Error during processing', 500

        name, ext = os.path.splitext(filename)
        scale_str = f"{outscale:.1f}".replace('.', '_')
        output_filename = f"{name}_{model_name}_s{scale_str}{'_face' if face_enhance else ''}{ext}"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        cv2.imwrite(output_path, output)

        return send_file(output_path, mimetype=f'image/{ext.lstrip(".")}')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)