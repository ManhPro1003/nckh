from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import math
import re
import fitz  # Thư viện PyMuPDF để đọc PDF

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- CÁC THUẬT TOÁN TẠO G-CODE ---
def generate_svg_grid_mm(w_mm, h_mm):
    grid = []
    for i in range(0, int(w_mm) + 1): grid.append(f'<line x1="{i}" y1="0" x2="{i}" y2="{h_mm}" stroke="#333" stroke-width="0.2"/>')
    for i in range(0, int(h_mm) + 1): grid.append(f'<line x1="0" y1="{i}" x2="{w_mm}" y2="{i}" stroke="#333" stroke-width="0.2"/>')
    for i in range(0, int(w_mm) + 1, 10):
        grid.append(f'<line x1="{i}" y1="0" x2="{i}" y2="{h_mm}" stroke="#555" stroke-width="0.5"/>')
        grid.append(f'<text x="{i + 1}" y="4" fill="#888" font-size="3" font-family="sans-serif">{i}</text>')
    for i in range(0, int(h_mm) + 1, 10):
        grid.append(f'<line x1="0" y1="{i}" x2="{w_mm}" y2="{i}" stroke="#555" stroke-width="0.5"/>')
        if i > 0: grid.append(f'<text x="1" y="{i - 1}" fill="#888" font-size="3" font-family="sans-serif">{i}</text>')
    return "".join(grid)

def process_pcb(gray_img, scale_x, scale_y, w_mm, h_mm, feedrate, thresh_val, pen_size_mm, z_up, z_down):
    h, w = gray_img.shape
    _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    gcode_lines = [f"G21\nG90\nG1 F{feedrate}\nG0 Z{z_up}\n"]
    svg_paths = []
    color = "#ffaa00"
    pen_px = max(1.0, pen_size_mm / scale_x) 

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.05, True) 
        if len(approx) > 2:
            pts = []
            sx, sy = approx[0][0]
            gx, gy = sx * scale_x, sy * scale_y
            gcode_lines.append(f"G0 Z{z_up}\nG0 X{gx:.2f} Y{h_mm - gy:.2f}\nG1 Z{z_down}")
            pts.append(f"{gx:.2f},{gy:.2f}") 
            for p in approx[1:]:
                x_mm, y_mm = p[0][0] * scale_x, p[0][1] * scale_y
                gcode_lines.append(f"G1 X{x_mm:.2f} Y{h_mm - y_mm:.2f}")
                pts.append(f"{x_mm:.2f},{y_mm:.2f}")
            gcode_lines.append(f"G1 X{gx:.2f} Y{h_mm - gy:.2f}")
            pts.append(f"{gx:.2f},{gy:.2f}")
            svg_paths.append(f'<polyline points="{" ".join(pts)}" stroke="{color}" fill="none" stroke-width="{pen_size_mm:.2f}" stroke-linejoin="round" stroke-linecap="round"/>')

    step_px = max(1, int(pen_px * 0.7)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fill_mask = cv2.erode(thresh, kernel, iterations=1)
    left_to_right = True 
    for y in range(step_px, h, step_px):
        row = fill_mask[y, :]
        segments = []
        start_x = -1
        for x in range(w):
            if row[x] == 255:
                if start_x == -1: start_x = x
            else:
                if start_x != -1:
                    if (x - start_x) > 1: segments.append((start_x, x - 1))
                    start_x = -1
        if start_x != -1 and (w - 1 - start_x) > 1: segments.append((start_x, w - 1))
        if not segments: continue
        if not left_to_right: segments.reverse()
        for seg in segments:
            x1, x2 = seg
            if not left_to_right: x1, x2 = x2, x1 
            gx1, gy1 = x1 * scale_x, y * scale_y
            gx2, gy2 = x2 * scale_x, y * scale_y
            gcode_lines.append(f"G0 Z{z_up}\nG0 X{gx1:.2f} Y{h_mm - gy1:.2f}\nG1 Z{z_down}")
            gcode_lines.append(f"G1 X{gx2:.2f} Y{h_mm - gy2:.2f}")
            svg_paths.append(f'<line x1="{gx1:.2f}" y1="{gy1:.2f}" x2="{gx2:.2f}" y2="{gy2:.2f}" stroke="{color}" stroke-width="{pen_size_mm:.2f}" stroke-linecap="round"/>')
        left_to_right = not left_to_right 
    gcode_lines.append(f"\nG0 Z{z_up}\nG0 X0 Y0")
    svg = f'<svg viewBox="0 0 {w_mm} {h_mm}" style="width: 100%; overflow: visible; background: #1a1a1a;">{generate_svg_grid_mm(w_mm, h_mm)}{"".join(svg_paths)}</svg>'
    return "\n".join(gcode_lines), svg

def process_edge(gray_img, scale_x, scale_y, w_mm, h_mm, feedrate, thresh_val, smooth_val, pen_size_mm, z_up, z_down):
    edges = cv2.Canny(gray_img, thresh_val, thresh_val * 2)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    gcode_lines = [f"G21\nG90\nG1 F{feedrate}\nG0 Z{z_up}\n"]
    svg_paths = []
    epsilon = float(smooth_val) / 10.0 
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) > 2:
            pts = []
            sx_mm, sy_mm = approx[0][0][0] * scale_x, approx[0][0][1] * scale_y
            gcode_lines.append(f"G0 Z{z_up}\nG0 X{sx_mm:.2f} Y{h_mm - sy_mm:.2f}\nG1 Z{z_down}")
            pts.append(f"{sx_mm:.2f},{sy_mm:.2f}")
            for p in approx[1:]:
                x_mm, y_mm = p[0][0] * scale_x, p[0][1] * scale_y
                gcode_lines.append(f"G1 X{x_mm:.2f} Y{h_mm - y_mm:.2f}")
                pts.append(f"{x_mm:.2f},{y_mm:.2f}")
            gcode_lines.append(f"G1 X{sx_mm:.2f} Y{h_mm - sy_mm:.2f}")
            pts.append(f"{sx_mm:.2f},{sy_mm:.2f}")
            svg_paths.append(f'<polyline points="{" ".join(pts)}" stroke="#00f2fe" fill="none" stroke-width="{pen_size_mm:.2f}" stroke-linejoin="round" stroke-linecap="round"/>')
            
    gcode_lines.append(f"\nG0 Z{z_up}\nG0 X0 Y0")
    svg = f'<svg viewBox="0 0 {w_mm} {h_mm}" style="width: 100%; overflow: visible; background: #1a1a1a;">{generate_svg_grid_mm(w_mm, h_mm)}{"".join(svg_paths)}</svg>'
    return "\n".join(gcode_lines), svg

def process_sketch(gray_img, scale_x, scale_y, w_mm, h_mm, feedrate, density, threshold_val, pen_size_mm, z_up, z_down):
    h, w = gray_img.shape
    spacing = max(2, int(12 - density / 10.0)) 
    gcode_lines = [f"G21\nG90\nG1 F{feedrate}\nG0 Z{z_up}\n"]
    svg_paths = []
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0) 
    def cast_rays(dx, dy, thresh):
        starts = []
        if dy == 1:
            for x in range(0, w, spacing): starts.append((x, 0))
            for y in range(spacing, h, spacing): starts.append((0, y))
        else:      
            for x in range(0, w, spacing): starts.append((x, h-1))
            for y in range(0, h-1, spacing): starts.append((0, y))
        for (sx, sy) in starts:
            in_seg, cx, cy, seg_start = False, sx, sy, None
            while 0 <= cx < w and 0 <= cy < h:
                if blurred[cy, cx] < thresh:
                    if not in_seg: in_seg, seg_start = True, (cx, cy)
                else:
                    if in_seg:
                        if math.dist(seg_start, (cx, cy)) > 3: add_line(seg_start[0], seg_start[1], cx, cy)
                        in_seg = False
                cx += dx; cy += dy
            if in_seg and math.dist(seg_start, (cx, cy)) > 3: add_line(seg_start[0], seg_start[1], cx, cy)
    def add_line(x1, y1, x2, y2):
        x1_mm, y1_mm = x1 * scale_x, y1 * scale_y
        x2_mm, y2_mm = x2 * scale_x, y2 * scale_y
        gcode_lines.append(f"G0 Z{z_up}\nG0 X{x1_mm:.2f} Y{h_mm - y1_mm:.2f}\nG1 Z{z_down}")
        gcode_lines.append(f"G1 X{x2_mm:.2f} Y{h_mm - y2_mm:.2f}")
        svg_paths.append(f'<line x1="{x1_mm:.2f}" y1="{y1_mm:.2f}" x2="{x2_mm:.2f}" y2="{y2_mm:.2f}" stroke="#ff007f" stroke-width="{pen_size_mm:.2f}" stroke-linecap="round"/>')
    cast_rays(1, 1, threshold_val)
    if threshold_val > 50: cast_rays(1, -1, threshold_val - 40)
    gcode_lines.append(f"\nG0 Z{z_up}\nG0 X0 Y0")
    svg = f'<svg viewBox="0 0 {w_mm} {h_mm}" style="width: 100%; overflow: visible; background: #1a1a1a;">{generate_svg_grid_mm(w_mm, h_mm)}{"".join(svg_paths)}</svg>'
    return "\n".join(gcode_lines), svg

def process_spiral(gray_img, scale_x, scale_y, w_mm, h_mm, feedrate, density, pen_size_mm, z_up, z_down):
    h, w = gray_img.shape
    cx, cy = w / 2, h / 2
    max_radius = min(cx, cy) - 2
    num_loops = max(10, int(density))
    gcode_lines = [f"G21\nG90\nG1 F{feedrate}\nG0 Z{z_up}\n"]
    svg_points = []
    theta, theta_step = 0, 0.05 
    is_first = True
    while True:
        r = max_radius * (theta / (num_loops * 2 * math.pi))
        if r > max_radius: break
        x_base, y_base = cx + r * math.cos(theta), cy + r * math.sin(theta)
        ix, iy = int(x_base), int(y_base)
        darkness = 1.0 - (gray_img[iy, ix] / 255.0) if (0 <= ix < w and 0 <= iy < h) else 0
        r_wiggled = r + ((max_radius / num_loops) * 0.95 * darkness) * math.sin(theta * 200)
        
        x_mm, y_mm = (cx + r_wiggled * math.cos(theta)) * scale_x, (cy + r_wiggled * math.sin(theta)) * scale_y
        svg_points.append(f"{x_mm:.2f},{y_mm:.2f}")
        
        if is_first:
            gcode_lines.append(f"G0 X{x_mm:.3f} Y{h_mm - y_mm:.3f}\nG1 Z{z_down}")
            is_first = False
        else:
            gcode_lines.append(f"G1 X{x_mm:.3f} Y{h_mm - y_mm:.3f}")
        theta += theta_step
    gcode_lines.append(f"\nG0 Z{z_up}\nG0 X0 Y0")
    points_str = " ".join(svg_points)
    svg = f'<svg viewBox="0 0 {w_mm} {h_mm}" style="width: 100%; overflow: visible; background: #1a1a1a;">{generate_svg_grid_mm(w_mm, h_mm)}<polyline points="{points_str}" stroke="#00ff00" fill="none" stroke-width="{pen_size_mm:.2f}" stroke-linejoin="round"/></svg>'
    return "\n".join(gcode_lines), svg

def process_image(file_path, main_mode, art_style, target_width_mm, target_height_mm, feedrate, thresh_val, smooth_val, density, pen_size_mm, z_up, z_down, g_scale):
    is_pdf = file_path.lower().endswith('.pdf')
    mm_per_px = None
    native_w_px = 0
    native_h_px = 0
    
    if is_pdf:
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        zoom = 4.0 
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        page_w_mm = page.rect.width * 25.4 / 72.0 
        
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 3: gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else: gray = img
        mm_per_px = page_w_mm / float(gray.shape[1])
        native_w_px = gray.shape[1]
        native_h_px = gray.shape[0]
    else:
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        if is_pdf and main_mode == 'pcb':
            cropped = gray[y:y+h, x:x+w]
            actual_width_mm = w * mm_per_px * (g_scale / 100.0)
            actual_height_mm = h * mm_per_px * (g_scale / 100.0)
            gray = cropped
        else:
            pad_x = int(w * 0.02)
            pad_y = int(h * 0.02)
            x_start = max(0, x - pad_x)
            y_start = max(0, y - pad_y)
            x_end = min(gray.shape[1], x + w + pad_x)
            y_end = min(gray.shape[0], y + h + pad_y)
            cropped = gray[y_start:y_end, x_start:x_end]
            gray = cropped
            
            actual_width_mm = target_width_mm * (g_scale / 100.0)
            actual_height_mm = actual_width_mm * (gray.shape[0] / gray.shape[1])
    else:
        actual_width_mm = target_width_mm * (g_scale / 100.0)
        actual_height_mm = actual_width_mm * (gray.shape[0] / gray.shape[1])
        
    scale_x = actual_width_mm / gray.shape[1]
    scale_y = actual_height_mm / gray.shape[0]
    
    if main_mode == 'pcb': 
        gcode, svg = process_pcb(gray, scale_x, scale_y, actual_width_mm, actual_height_mm, feedrate, thresh_val, pen_size_mm, z_up, z_down)
    elif main_mode == 'art' and art_style == 'spiral': 
        gcode, svg = process_spiral(gray, scale_x, scale_y, actual_width_mm, actual_height_mm, feedrate, density, pen_size_mm, z_up, z_down)
    elif main_mode == 'art' and art_style == 'sketch': 
        gcode, svg = process_sketch(gray, scale_x, scale_y, actual_width_mm, actual_height_mm, feedrate, density, thresh_val, pen_size_mm, z_up, z_down)
    else:
        gcode, svg = process_edge(gray, scale_x, scale_y, actual_width_mm, actual_height_mm, feedrate, thresh_val, smooth_val, pen_size_mm, z_up, z_down)
        
    return gcode, svg, actual_width_mm, actual_height_mm, native_w_px, native_h_px

@app.route('/')
def index(): return render_template('web.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files: return jsonify({'error': 'Chưa chọn file'})
    file = request.files['image']
    mode, style = request.form.get('main_mode', 'art'), request.form.get('art_style', 'edge')
    
    width = float(request.form.get('target_width', 100))
    height = float(request.form.get('target_height', 0))
    
    thresh = int(request.form.get('threshold', 127))
    smooth = float(request.form.get('smoothing', 1))
    dens = float(request.form.get('density', 30))
    pen_size = float(request.form.get('pen_size', 0.3))
    feed = int(request.form.get('feedrate', 1500))
    z_up = float(request.form.get('z_up', 5.0))
    z_down = float(request.form.get('z_down', 0.0))
    g_scale = float(request.form.get('g_scale', 100.0))
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    gcode, svg, calc_w, calc_h, px_w, px_h = process_image(filepath, mode, style, width, height, feed, thresh, smooth, dens, pen_size, z_up, z_down, g_scale)
    
    return jsonify({
        'gcode': gcode, 
        'svg': svg,
        'calc_w': calc_w, 
        'calc_h': calc_h,
        'native_px_w': px_w,
        'native_px_h': px_h
    })

# Cổng 7860 là cổng mặc định của Hugging Face Spaces
if __name__ == '__main__': app.run(debug=False, port=7860, host="0.0.0.0")
