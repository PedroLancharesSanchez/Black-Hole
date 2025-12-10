from PIL import Image, ImageDraw
import os
import random

# Create output directory
output_dir = "test_images"
# Clean up existing directory if needed or just ensure it exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Image size
img_size = 200

def get_random_color():
    """Generate a random RGB tuple"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex string"""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def create_circle(color, index, used_colors_set):
    """Create a circle image"""
    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw circle
    margin = 20
    draw.ellipse([margin, margin, img_size - margin, img_size - margin], 
                 fill=color, outline=color)
    
    hex_color = rgb_to_hex(color).replace('#', '')
    filename = f"{output_dir}/circle_{hex_color}_{index:02d}.png"
    img.save(filename)
    return filename

def create_square(color, index, used_colors_set):
    """Create a square image"""
    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw square
    margin = 30
    draw.rectangle([margin, margin, img_size - margin, img_size - margin], 
                   fill=color, outline=color)
    
    hex_color = rgb_to_hex(color).replace('#', '')
    filename = f"{output_dir}/square_{hex_color}_{index:02d}.png"
    img.save(filename)
    return filename

def create_triangle(color, index, used_colors_set):
    """Create a triangle image"""
    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw equilateral triangle (pointing up)
    margin = 20
    points = [
        (img_size // 2, margin),  # Top vertex
        (margin, img_size - margin),  # Bottom left
        (img_size - margin, img_size - margin)  # Bottom right
    ]
    draw.polygon(points, fill=color, outline=color)
    
    hex_color = rgb_to_hex(color).replace('#', '')
    filename = f"{output_dir}/triangle_{hex_color}_{index:02d}.png"
    img.save(filename)
    return filename

# Generate images
print("Generating test images...")
print(f"Creating 20 images of each shape (circles, squares, triangles)")
print(f"Total: 60 images\n")

count = 0
used_colors = set()

def get_unique_color(used_set):
    while True:
        c = get_random_color()
        if c not in used_set:
            used_set.add(c)
            return c

# Generate 20 of each shape
for i in range(20):
    color = get_unique_color(used_colors)
    create_circle(color, i + 1, used_colors)
    count += 1

for i in range(20):
    color = get_unique_color(used_colors)
    create_square(color, i + 1, used_colors)
    count += 1

for i in range(20):
    color = get_unique_color(used_colors)
    create_triangle(color, i + 1, used_colors)
    count += 1

print(f"\n✓ Generated {count} images successfully!")
print(f"Images saved in: {output_dir}/")
