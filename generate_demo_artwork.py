import os
import cv2
import numpy as np

def generate_synthetic_content_landscape(output_path="input/content_landscape.jpg", width=640, height=480):
    """Generates a synthetic photograph landscape content image (Mountain, Sun, Lake)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = np.ones((height, width, 3), dtype=np.uint8) * 220
    
    # Sky Gradient (Blue to Orange)
    for y in range(int(height * 0.55)):
        img[y, :] = (int(240 - y*0.3), int(180 - y*0.2), int(100 + y*0.2))
        
    # Sun
    cv2.circle(img, (int(width * 0.75), int(height * 0.3)), 40, (200, 240, 255), -1)
    
    # Mountain Silhouettes
    pts1 = np.array([[0, height], [int(width*0.35), int(height*0.25)], [int(width*0.7), height]], dtype=np.int32)
    cv2.fillPoly(img, [pts1], (70, 70, 70))
    
    pts2 = np.array([[int(width*0.4), height], [int(width*0.8), int(height*0.35)], [width, height]], dtype=np.int32)
    cv2.fillPoly(img, [pts2], (40, 40, 40))
    
    # Water Lake (Bottom 30%)
    img[int(height*0.7):, :] = (180, 120, 50)
    
    cv2.putText(img, "CONTENT: MOUNTAIN LANDSCAPE", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    cv2.imwrite(output_path, img)
    print(f"[OK] Synthetic content image saved to '{output_path}'")
    return output_path

def generate_synthetic_style_starry(output_path="input/style_starry_night.jpg", width=640, height=480):
    """Generates a synthetic painterly style reference image (Van Gogh Starry Swirls)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    style = np.ones((height, width, 3), dtype=np.uint8) * 120
    style[:] = (140, 60, 20) # Deep Cobalt Blue
    
    # Draw Yellow Swirling Impasto Strokes
    for r in range(40, 250, 35):
        cv2.ellipse(style, (width // 2, height // 2), (r, int(r * 0.6)), 30, 0, 270, (0, 220, 255), 8, lineType=cv2.LINE_AA)
        cv2.ellipse(style, (width // 2 + 80, height // 2 - 40), (r, int(r * 0.5)), -20, 0, 300, (50, 180, 240), 6, lineType=cv2.LINE_AA)
        
    # Star Orbs
    cv2.circle(style, (120, 100), 30, (50, 240, 255), -1)
    cv2.circle(style, (width - 140, 120), 40, (50, 240, 255), -1)
    
    cv2.putText(style, "STYLE: VANGOGH STARRY NIGHT", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    cv2.imwrite(output_path, style)
    print(f"[OK] Synthetic style image saved to '{output_path}'")
    return output_path

def generate_all_artwork():
    """Generates content and style image samples."""
    generate_synthetic_content_landscape("input/content_landscape.jpg")
    generate_synthetic_style_starry("input/style_starry_night.jpg")

if __name__ == "__main__":
    generate_all_artwork()
