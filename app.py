import os
import cv2
import torch
import glob
import uuid
import numpy as np
from flask import Flask, request, send_file, render_template, jsonify
from werkzeug.utils import secure_filename
try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    psnr = None
    ssim = None

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
analysis_cache = {}


def calculate_sharpness(image):
    """计算图像清晰度（拉普拉斯方差）"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def calculate_contrast(image):
    """计算图像对比度（RMS对比度）"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    mean = np.mean(gray)
    rms_contrast = np.sqrt(np.mean((gray - mean) ** 2))
    return float(rms_contrast / 255.0)


def calculate_noise_level(image):
    """计算图像噪声水平（拉普拉斯噪声估计）"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 使用高斯模糊估计噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    noise = gray.astype(np.float32) - blurred.astype(np.float32)
    noise_level = np.std(noise) / 255.0
    return float(noise_level)


def calculate_edge_density(image):
    """计算边缘密度"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    return float(edge_density)


def analyze_color_distribution(image):
    """分析色彩分布"""
    if len(image.shape) == 3:
        # 分别计算RGB通道的均值和标准差
        mean_bgr = np.mean(image, axis=(0, 1))
        std_bgr = np.std(image, axis=(0, 1))
        brightness = float(np.mean(mean_bgr))
        color_variance = float(np.mean(std_bgr))
    else:
        brightness = float(np.mean(image))
        color_variance = 0.0

    return {
        'brightness': brightness / 255.0,
        'color_variance': color_variance / 255.0
    }


def analyze_image(image, file_path=None):
    """全面分析图像"""
    # 获取元数据
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0

    # 计算各项指标
    metrics = {
        'sharpness': calculate_sharpness(image),
        'contrast': calculate_contrast(image),
        'noise_level': calculate_noise_level(image),
        'edge_density': calculate_edge_density(image)
    }

    # 添加色彩分析
    metrics.update(analyze_color_distribution(image))

    return {
        'metadata': {
            'width': int(width),
            'height': int(height),
            'channels': int(channels),
            'file_size': int(file_size),
            'color_space': 'BGR' if channels == 3 else 'Grayscale'
        },
        'metrics': metrics
    }


def calculate_quality_metrics(original_img, result_img):
    """计算图像质量评估指标"""
    if psnr is None or ssim is None:
        return {'psnr': None, 'ssim': None}

    try:
        # 确保图像尺寸一致
        if original_img.shape != result_img.shape:
            result_img = cv2.resize(result_img, (original_img.shape[1], original_img.shape[0]))

        # 转换为灰度图像计算SSIM
        if len(original_img.shape) == 3:
            orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original_img
            result_gray = result_img

        # 计算PSNR和SSIM
        psnr_value = psnr(original_img, result_img, data_range=255)
        ssim_value = ssim(orig_gray, result_gray, data_range=255)

        return {
            'psnr': float(psnr_value),
            'ssim': float(ssim_value)
        }
    except Exception as e:
        print(f"Error calculating quality metrics: {e}")
        return {'psnr': None, 'ssim': None}


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
    return_analysis = request.form.get('include_analysis', '').lower() == 'true'

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None: return 'Cannot read input image', 500

        # 生成请求ID
        request_id = str(uuid.uuid4())

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

        # 进行图像分析
        original_analysis = analyze_image(img, input_path)
        result_analysis = analyze_image(output, output_path)
        quality_metrics = calculate_quality_metrics(img, output)

        # 计算改进百分比
        improvement = {}
        if original_analysis['metrics']['sharpness'] > 0:
            improvement['sharpness_gain'] = f"+{((result_analysis['metrics']['sharpness'] / original_analysis['metrics']['sharpness'] - 1) * 100):.1f}%"
        if original_analysis['metrics']['contrast'] > 0:
            improvement['contrast_gain'] = f"+{((result_analysis['metrics']['contrast'] / original_analysis['metrics']['contrast'] - 1) * 100):.1f}%"
        if original_analysis['metrics']['noise_level'] > 0:
            improvement['noise_reduction'] = f"{((result_analysis['metrics']['noise_level'] / original_analysis['metrics']['noise_level'] - 1) * 100):.1f}%"

        # 构建完整分析结果
        analysis_result = {
            'request_id': request_id,
            'processing_info': {
                'model': model_name,
                'scale_factor': outscale,
                'face_enhance': face_enhance
            },
            'original': original_analysis,
            'result': {
                **result_analysis,
                'quality_scores': quality_metrics
            },
            'improvement': improvement
        }

        # 缓存分析结果
        analysis_cache[request_id] = analysis_result

        # 总是进行分析，但根据请求类型返回不同格式
        if return_analysis:
            # 返回包含分析结果和图片base64的JSON
            import base64
            with open(output_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            analysis_result['image_data'] = f"data:image/{ext.lstrip('.')};base64,{img_base64}"
            return jsonify(analysis_result)

        return send_file(output_path, mimetype=f'image/{ext.lstrip(".")}')


@app.route('/analysis/<request_id>', methods=['GET'])
def get_analysis(request_id):
    """获取指定请求的分析结果"""
    if request_id in analysis_cache:
        return jsonify(analysis_cache[request_id])
    else:
        return jsonify({'error': 'Analysis not found'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
