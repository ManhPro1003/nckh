from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import math
import re

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

def process_pcb(gray_img, scale, target_width_mm, feedrate, thresh_val, pen_size_mm, z_up, z_down):
    h, w = gray_img.shape
    h_mm, w_mm = h * scale, w * scale
    _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    gcode_lines = [f"G21\nG90\nG1 F{feedrate}\nG0 Z{z_up}\n"]
    svg_paths = []
    color = "#ffaa00"
    pen_px = max(1.0, pen_size_mm / scale) 

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.05, True) 
        if len(approx) > 2:
            pts = []
            sx, sy = approx[0][0]
            gx, gy = sx * scale, (h - sy) * scale
            gcode_lines.append(f"G0 Z{z_up}\nG0 X{gx:.2f} Y{gy:.2f}\nG1 Z{z_down}")
            pts.append(f"{gx:.2f},{h_mm - gy:.2f}") 
            for p in approx[1:]:
                x, y = p[0]
                x_mm, y_mm = x * scale, y * scale
                gcode_lines.append(f"G1 X{x_mm:.2f} Y{h_mm - y_mm:.2f}")
                pts.append(f"{x_mm:.2f},{y_mm:.2f}")
            gcode_lines.append(f"G1 X{gx:.2f} Y{gy:.2f}")
            pts.append(f"{gx:.2f},{h_mm - gy:.2f}")
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
            gx1, gy1 = x1 * scale, (h - y) * scale
            gx2, gy2 = x2 * scale, (h - y) * scale
            gcode_lines.append(f"G0 Z{z_up}\nG0 X{gx1:.2f} Y{gy1:.2f}\nG1 Z{z_down}")
            gcode_lines.append(f"G1 X{gx2:.2f} Y{gy2:.2f}")
            svg_paths.append(f'<line x1="{gx1:.2f}" y1="{h_mm - gy1:.2f}" x2="{gx2:.2f}" y2="{h_mm - gy2:.2f}" stroke="{color}" stroke-width="{pen_size_mm:.2f}" stroke-linecap="round"/>')
        left_to_right = not left_to_right 
    gcode_lines.append(f"\nG0 Z{z_up}\nG0 X0 Y0")
    svg = f'<svg viewBox="0 0 {w_mm} {h_mm}" style="width: 100%; overflow: visible; background: #1a1a1a;">{generate_svg_grid_mm(w_mm, h_mm)}{"".join(svg_paths)}</svg>'
    return "\n".join(gcode_lines), svg

def process_edge(gray_img, scale, target_width_mm, feedrate, thresh_val, smooth_val, pen_size_mm, z_up, z_down):
    h, w = gray_img.shape
    h_mm, w_mm = h * scale, w * scale
    _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gcode_lines = [f"G21\nG90\nG1 F{feedrate}\nG0 Z{z_up}\n"]
    svg_paths = []
    epsilon = float(smooth_val) / 10.0 
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) > 2:
            pts = []
            sx_mm, sy_mm = approx[0][0][0] * scale, approx[0][0][1] * scale
            gcode_lines.append(f"G0 Z{z_up}\nG0 X{sx_mm:.2f} Y{h_mm - sy_mm:.2f}\nG1 Z{z_down}")
            pts.append(f"{sx_mm:.2f},{sy_mm:.2f}")
            for p in approx[1:]:
                x_mm, y_mm = p[0][0] * scale, p[0][1] * scale
                gcode_lines.append(f"G1 X{x_mm:.2f} Y{h_mm - y_mm:.2f}")
                pts.append(f"{x_mm:.2f},{y_mm:.2f}")
            gcode_lines.append(f"G1 X{sx_mm:.2f} Y{h_mm - sy_mm:.2f}")
            pts.append(f"{sx_mm:.2f},{sy_mm:.2f}")
            svg_paths.append(f'<polyline points="{" ".join(pts)}" stroke="#00f2fe" fill="none" stroke-width="{pen_size_mm:.2f}" stroke-linejoin="round" stroke-linecap="round"/>')
    gcode_lines.append(f"\nG0 Z{z_up}\nG0 X0 Y0")
    svg = f'<svg viewBox="0 0 {w_mm} {h_mm}" style="width: 100%; overflow: visible; background: #1a1a1a;">{generate_svg_grid_mm(w_mm, h_mm)}{"".join(svg_paths)}</svg>'
    return "\n".join(gcode_lines), svg

def process_sketch(gray_img, scale, target_width_mm, feedrate, density, threshold_val, pen_size_mm, z_up, z_down):
    h, w = gray_img.shape
    h_mm, w_mm = h * scale, w * scale
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
        x1_mm, y1_mm = x1 * scale, y1 * scale
        x2_mm, y2_mm = x2 * scale, y2 * scale
        gcode_lines.append(f"G0 Z{z_up}\nG0 X{x1_mm:.2f} Y{h_mm - y1_mm:.2f}\nG1 Z{z_down}")
        gcode_lines.append(f"G1 X{x2_mm:.2f} Y{h_mm - y2_mm:.2f}")
        svg_paths.append(f'<line x1="{x1_mm:.2f}" y1="{y1_mm:.2f}" x2="{x2_mm:.2f}" y2="{y2_mm:.2f}" stroke="#ff007f" stroke-width="{pen_size_mm:.2f}" stroke-linecap="round"/>')
    cast_rays(1, 1, threshold_val)
    if threshold_val > 50: cast_rays(1, -1, threshold_val - 40)
    gcode_lines.append(f"\nG0 Z{z_up}\nG0 X0 Y0")
    svg = f'<svg viewBox="0 0 {w_mm} {h_mm}" style="width: 100%; overflow: visible; background: #1a1a1a;">{generate_svg_grid_mm(w_mm, h_mm)}{"".join(svg_paths)}</svg>'
    return "\n".join(gcode_lines), svg

def process_spiral(gray_img, scale, target_width_mm, feedrate, density, pen_size_mm, z_up, z_down):
    h, w = gray_img.shape
    h_mm, w_mm = h * scale, w * scale
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
        x_mm, y_mm = (cx + r_wiggled * math.cos(theta)) * scale, (cy + r_wiggled * math.sin(theta)) * scale
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

def process_image(img_path, main_mode, art_style, target_width_mm, feedrate, thresh_val, smooth_val, density, pen_size_mm, z_up, z_down, g_scale):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # === TÍNH NĂNG MỚI: TỰ ĐỘNG CẮT PHẦN THỪA (AUTO-CROP) ===
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad_x = int(w * 0.02)
        pad_y = int(h * 0.02)
        x_start = max(0, x - pad_x)
        y_start = max(0, y - pad_y)
        x_end = min(gray.shape[1], x + w + pad_x)
        y_end = min(gray.shape[0], y + h + pad_y)
        gray = gray[y_start:y_end, x_start:x_end]
    # =======================================================
        
    actual_width_mm = target_width_mm * (g_scale / 100.0)
    scale = actual_width_mm / gray.shape[1]
    
    if main_mode == 'pcb': return process_pcb(gray, scale, actual_width_mm, feedrate, thresh_val, pen_size_mm, z_up, z_down)
    if main_mode == 'art' and art_style == 'spiral': return process_spiral(gray, scale, actual_width_mm, feedrate, density, pen_size_mm, z_up, z_down)
    if main_mode == 'art' and art_style == 'sketch': return process_sketch(gray, scale, actual_width_mm, feedrate, density, thresh_val, pen_size_mm, z_up, z_down)
    return process_edge(gray, scale, actual_width_mm, feedrate, thresh_val, smooth_val, pen_size_mm, z_up, z_down)

@app.route('/')
def index(): return render_template('web.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files: return jsonify({'error': 'Chưa chọn ảnh'})
    file = request.files['image']
    mode, style = request.form.get('main_mode', 'art'), request.form.get('art_style', 'edge')
    width = float(request.form.get('target_width', 100))
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
    gcode, svg = process_image(filepath, mode, style, width, feed, thresh, smooth, dens, pen_size, z_up, z_down, g_scale)
    return jsonify({'gcode': gcode, 'svg': svg})

if __name__ == '__main__': app.run(debug=True, port=7860, host="0.0.0.0")
