import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
from PIL import Image, ImageDraw
import io
import base64
import time
from matplotlib.animation import FuncAnimation
import imageio
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Abstract Art Generator",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .title {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 50px;
        font-weight: bold;
        letter-spacing: -1px;
        text-align: center;
        margin-bottom: 30px;
    }
    .subtitle {
        color: #aaaaaa;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 20px;
        text-align: center;
        margin-bottom: 40px;
    }
    .stButton>button {
        background-color: #4c566a;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #5e81ac;
    }
</style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("<h1 class='title'>Abstract Art Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Create unique algorithmic artworks with a click</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Art Controls")
    
    # Art Style Selection
    art_style = st.selectbox(
        "Choose Art Style",
        ["Fractal Flow", "Particle System", "Wave Interference", "Color Field", "Cellular Automaton"]
    )
    
    # Color Palette
    st.subheader("Color Palette")
    color_scheme = st.selectbox(
        "Select a color scheme",
        ["Vivid", "Pastel", "Monochrome", "Earth", "Neon", "Ocean", "Random"]
    )
    
    # Resolution
    resolution = st.slider("Resolution", min_value=100, max_value=800, value=500, step=100)
    
    # Complexity
    complexity = st.slider("Complexity", min_value=1, max_value=10, value=5, step=1)
    
    # Randomness
    randomness = st.slider("Randomness", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    # Seed for reproducibility
    use_random_seed = st.checkbox("Use random seed", value=True)
    if not use_random_seed:
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)
    else:
        seed = random.randint(0, 9999)
        st.write(f"Random seed: {seed}")
    
    # Animation Options
    st.subheader("Animation Options")
    enable_animation = st.checkbox("Enable Animation (GIF)", value=False)
    if enable_animation:
        animation_frames = st.slider("Animation Frames", min_value=5, max_value=30, value=15, step=5)
        animation_speed = st.slider("Animation Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        animation_type = st.selectbox(
            "Animation Type",
            ["Parameter Shift", "Color Shift", "Zoom Effect"]
        )
    
    # Live Animation Option
    live_animation = st.checkbox("Enable Live Animation", value=False, 
                               help="Continuously change the artwork in real-time")
    
    if live_animation and enable_animation:
        st.warning("Please choose either GIF animation or live animation, not both.")
    
    # Generate button
    generate_button = st.button("Generate Artwork", use_container_width=True)

# Define color palettes
def get_color_palette(scheme):
    palettes = {
        "Vivid": ["#FF5E5B", "#D8D8D8", "#FFFFEA", "#00CECB", "#FFED66"],
        "Pastel": ["#A0D2EB", "#E5EAF5", "#D0BDF4", "#8458B3", "#A28089"],
        "Monochrome": ["#0D0D0D", "#262626", "#595959", "#A6A6A6", "#F2F2F2"],
        "Earth": ["#5E3929", "#CD8B76", "#FBBF45", "#1A936F", "#114B5F"],
        "Neon": ["#FF00FF", "#00FFFF", "#FF9933", "#33FF33", "#FF3366"],
        "Ocean": ["#05445E", "#189AB4", "#75E6DA", "#D4F1F9", "#003459"]
    }
    
    if scheme == "Random":
        # Generate random colors
        return ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for _ in range(5)]
    
    return palettes[scheme]

# Create custom colormaps from palettes
def create_colormap(palette):
    # Normalize the positions
    positions = np.linspace(0, 1, len(palette))
    
    # Create a dictionary for the colormap
    colors_dict = {
        'red': [],
        'green': [],
        'blue': []
    }
    
    # Convert hex to RGB and populate the dictionary
    for pos, hex_color in zip(positions, palette):
        # Convert hex to RGB
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
        
        colors_dict['red'].append((pos, rgb[0], rgb[0]))
        colors_dict['green'].append((pos, rgb[1], rgb[1]))
        colors_dict['blue'].append((pos, rgb[2], rgb[2]))
    
    return LinearSegmentedColormap('custom_cmap', colors_dict)

# Fractal Flow generator
def create_fractal_flow(size, palette, complexity, randomness, seed):
    np.random.seed(seed)
    
    # Create a grid of coordinates
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Initialize the image data
    data = np.zeros((size, size))
    
    # Ensure complexity is a valid number
    try:
        iterations = int(float(complexity) * 10)
    except (ValueError, TypeError):
        # Default to a safe value if complexity is invalid
        iterations = 10
    
    # Apply fractal algorithm
    for i in range(iterations):
        # Create random coefficients
        a = np.random.uniform(-2, 2) * randomness
        b = np.random.uniform(-2, 2) * randomness
        c = np.random.uniform(-2, 2) * randomness
        d = np.random.uniform(-2, 2) * randomness
        e = np.random.uniform(-2, 2) * randomness
        f = np.random.uniform(-2, 2) * randomness
        
        # Apply transformation
        Z = X**2 * a + Y**2 * b + X * Y * c + X * d + Y * e + f
        
        # Combine with the existing data
        data += np.sin(Z * np.pi * (i+1)/10) * np.random.uniform(0.1, 0.5)
    
    # Normalize the data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Create a colormap from the palette
    cmap = create_colormap(palette)
    
    # Create a figure and plot
    fig, ax = plt.figure(figsize=(10, 10), dpi=100), plt.Axes(plt.figure(figsize=(10, 10), dpi=100), [0., 0., 1., 1.])
    ax.set_axis_off()
    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(data, cmap=cmap, aspect='equal')
    plt.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

# Particle System generator
def create_particle_system(size, palette, complexity, randomness, seed):
    np.random.seed(seed)
    
    # Create a blank image
    img = Image.new('RGB', (size, size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Convert palette hex colors to RGB
    colors = [tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for color in palette]
    
    # Number of particles
    num_particles = complexity * 500
    
    # Generate particles
    for _ in range(num_particles):
        # Initial position
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        
        # Particle size
        radius = np.random.randint(1, 5 + complexity)
        
        # Color
        color = random.choice(colors)
        
        # Draw the particle
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        
        # Add some connecting lines with probability based on randomness
        if np.random.random() < randomness:
            # Find another random point
            x2 = np.random.randint(0, size)
            y2 = np.random.randint(0, size)
            
            # Draw a line if not too far away
            dist = np.sqrt((x2 - x)**2 + (y2 - y)**2)
            if dist < size / 4:
                # Fade the color based on distance
                alpha = int(255 * (1 - dist / (size / 4)))
                line_color = tuple(max(0, min(255, int(c * alpha / 255))) for c in color)
                draw.line((x, y, x2, y2), fill=line_color, width=1)
    
    return img

# Wave Interference generator
def create_wave_interference(size, palette, complexity, randomness, seed):
    np.random.seed(seed)
    
    # Create a grid of coordinates
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Initialize the wave data
    data = np.zeros((size, size))
    
    # Generate wave sources
    num_sources = complexity
    for _ in range(num_sources):
        # Random position for wave source
        source_x = np.random.uniform(-5, 5)
        source_y = np.random.uniform(-5, 5)
        
        # Calculate distance from each point to the source
        distance = np.sqrt((X - source_x)**2 + (Y - source_y)**2)
        
        # Wave amplitude and frequency (with randomness)
        amplitude = np.random.uniform(0.5, 1.0)
        frequency = np.random.uniform(0.5, 1.5) * (1 + randomness)
        phase = np.random.uniform(0, 2*np.pi)
        
        # Add this wave to the data
        data += amplitude * np.sin(distance * frequency + phase)
    
    # Normalize the data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Create a colormap from the palette
    cmap = create_colormap(palette)
    
    # Create a figure and plot
    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(data, cmap=cmap)
    plt.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

# Color Field generator
def create_color_field(size, palette, complexity, randomness, seed):
    np.random.seed(seed)
    
    # Create blank image
    img = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Convert hex colors to RGB
    colors = [tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for color in palette]
    
    # Number of shapes
    num_shapes = complexity * 20
    
    # Draw color fields
    for _ in range(num_shapes):
        # Random shape parameters
        x0 = np.random.randint(-size//2, size)
        y0 = np.random.randint(-size//2, size)
        width = np.random.randint(size//10, size//2)
        height = np.random.randint(size//10, size//2)
        
        # Add randomness to dimensions
        if np.random.random() < randomness:
            width = int(width * np.random.uniform(0.5, 1.5))
            height = int(height * np.random.uniform(0.5, 1.5))
        
        # Choose shape type (rectangle, ellipse)
        shape_type = np.random.choice(['rectangle', 'ellipse'])
        
        # Color with random alpha for blending
        color = random.choice(colors)
        alpha = np.random.uniform(0.2, 0.8)
        
        # Convert to RGBA for alpha blending
        fill_color = color + (int(alpha * 255),)
        
        # Draw the shape
        if shape_type == 'rectangle':
            # Rectangles can be rotated for more interest
            angle = np.random.uniform(0, 360) * randomness
            rect = img.copy()
            rect_draw = ImageDraw.Draw(rect)
            rect_draw.rectangle([x0, y0, x0+width, y0+height], fill=color)
            rect = rect.rotate(angle, center=(x0+width//2, y0+height//2), resample=Image.BICUBIC)
            img = Image.alpha_composite(img.convert('RGBA'), rect.convert('RGBA'))
            draw = ImageDraw.Draw(img)
        else:
            draw.ellipse([x0, y0, x0+width, y0+height], fill=color)
    
    # Convert back to RGB
    img = img.convert('RGB')
    
    return img

# Cellular Automaton generator
def create_cellular_automaton(size, palette, complexity, randomness, seed):
    np.random.seed(seed)
    
    # Create initial random state
    grid = np.random.choice([0, 1], size=(size, size), p=[1-randomness, randomness])
    
    # Define rule based on complexity
    # Higher complexity = more complex rules
    rule_num = (complexity * 10) % 256
    rule = [(rule_num >> i) & 1 for i in range(8)]
    
    # Evolve the cellular automaton
    for _ in range(size // 2):
        new_grid = np.zeros_like(grid)
        for i in range(size):
            for j in range(size):
                # Get neighborhood (with wrapping)
                left = grid[i, (j-1) % size]
                center = grid[i, j]
                right = grid[i, (j+1) % size]
                
                # Compute rule index
                idx = (left << 2) | (center << 1) | right
                
                # Apply rule
                new_grid[(i+1) % size, j] = rule[idx]
        
        grid = new_grid
    
    # Convert binary grid to colors
    data = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Convert palette hex colors to RGB
    colors = [tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for color in palette]
    
    # Assign colors based on patterns in the grid
    for i in range(size):
        for j in range(size):
            # Find patterns in 3x3 neighborhood
            pattern_sum = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = (i+di) % size, (j+dj) % size
                    pattern_sum += grid[ni, nj]
            
            # Map pattern to color
            color_idx = pattern_sum % len(colors)
            data[i, j] = colors[color_idx]
    
    # Create image from array
    img = Image.fromarray(data)
    
    return img

# Animation Functions

# Animated Fractal Flow generator
def create_animated_fractal_flow(size, palette, complexity, randomness, seed, frames=20, animation_type="Parameter Shift", animation_speed=0.5):
    np.random.seed(seed)
    images = []
    
    # Calculate frame duration (in ms) based on animation speed
    frame_duration = int(100 / animation_speed)  # Faster speed = shorter duration
    
    # Generate frames
    for frame in range(frames):
        # Modify parameters based on animation type
        if animation_type == "Parameter Shift":
            # Gradually shift parameters
            frame_complexity = complexity * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi))
            frame_randomness = randomness * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi + np.pi/4))
            frame_palette = palette
            frame_seed = seed
        
        elif animation_type == "Color Shift":
            # Shift colors in the palette
            frame_complexity = complexity
            frame_randomness = randomness
            
            # Convert hex colors to RGB, shift hue, and convert back to hex
            frame_palette = []
            for color in palette:
                # Convert hex to RGB
                r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                # Simple color shift (rotate RGB values)
                shift_amount = frame / frames
                r_new = int((r + shift_amount * 255) % 255)
                g_new = int((g + shift_amount * 255) % 255)
                b_new = int((b + shift_amount * 255) % 255)
                
                # Convert back to hex
                frame_palette.append(f'#{r_new:02x}{g_new:02x}{b_new:02x}')
            
            frame_seed = seed
        
        elif animation_type == "Zoom Effect":
            # Keep same parameters but change seed for a "zoom" effect
            frame_complexity = complexity
            frame_randomness = randomness
            frame_palette = palette
            frame_seed = seed + frame  # Change seed for zoom effect
        
        # Create the frame
        img = create_fractal_flow(size, frame_palette, frame_complexity, frame_randomness, frame_seed)
        images.append(img)
    
    # Create a temporary file for the GIF
    temp_file = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    
    # Save the frames as a GIF
    images[0].save(
        temp_file.name,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=frame_duration,
        loop=0
    )
    
    return temp_file.name

# Animated Particle System generator
def create_animated_particle_system(size, palette, complexity, randomness, seed, frames=20, animation_type="Parameter Shift", animation_speed=0.5):
    np.random.seed(seed)
    images = []
    
    # Calculate frame duration (in ms) based on animation speed
    frame_duration = int(100 / animation_speed)  # Faster speed = shorter duration
    
    # Generate frames
    for frame in range(frames):
        # Modify parameters based on animation type
        if animation_type == "Parameter Shift":
            # Gradually shift parameters
            frame_complexity = complexity * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi))
            frame_randomness = randomness * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi + np.pi/4))
            frame_palette = palette
            frame_seed = seed
        
        elif animation_type == "Color Shift":
            # Shift colors in the palette
            frame_complexity = complexity
            frame_randomness = randomness
            
            # Convert hex colors to RGB, shift hue, and convert back to hex
            frame_palette = []
            for color in palette:
                # Convert hex to RGB
                r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                # Simple color shift (rotate RGB values)
                shift_amount = frame / frames
                r_new = int((r + shift_amount * 255) % 255)
                g_new = int((g + shift_amount * 255) % 255)
                b_new = int((b + shift_amount * 255) % 255)
                
                # Convert back to hex
                frame_palette.append(f'#{r_new:02x}{g_new:02x}{b_new:02x}')
            
            frame_seed = seed
        
        elif animation_type == "Zoom Effect":
            # Keep same parameters but change seed for a "zoom" effect
            frame_complexity = complexity
            frame_randomness = randomness
            frame_palette = palette
            frame_seed = seed + frame  # Change seed for zoom effect
        
        # Create the frame
        img = create_particle_system(size, frame_palette, frame_complexity, frame_randomness, frame_seed)
        images.append(img)
    
    # Create a temporary file for the GIF
    temp_file = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    
    # Save the frames as a GIF
    images[0].save(
        temp_file.name,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=frame_duration,
        loop=0
    )
    
    return temp_file.name

# Animated Wave Interference generator
def create_animated_wave_interference(size, palette, complexity, randomness, seed, frames=20, animation_type="Parameter Shift", animation_speed=0.5):
    np.random.seed(seed)
    images = []
    
    # Calculate frame duration (in ms) based on animation speed
    frame_duration = int(100 / animation_speed)  # Faster speed = shorter duration
    
    # Generate frames
    for frame in range(frames):
        # Modify parameters based on animation type
        if animation_type == "Parameter Shift":
            # Gradually shift parameters
            frame_complexity = complexity * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi))
            frame_randomness = randomness * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi + np.pi/4))
            frame_palette = palette
            frame_seed = seed
        
        elif animation_type == "Color Shift":
            # Shift colors in the palette
            frame_complexity = complexity
            frame_randomness = randomness
            
            # Convert hex colors to RGB, shift hue, and convert back to hex
            frame_palette = []
            for color in palette:
                # Convert hex to RGB
                r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                # Simple color shift (rotate RGB values)
                shift_amount = frame / frames
                r_new = int((r + shift_amount * 255) % 255)
                g_new = int((g + shift_amount * 255) % 255)
                b_new = int((b + shift_amount * 255) % 255)
                
                # Convert back to hex
                frame_palette.append(f'#{r_new:02x}{g_new:02x}{b_new:02x}')
            
            frame_seed = seed
        
        elif animation_type == "Zoom Effect":
            # Keep same parameters but change seed for a "zoom" effect
            frame_complexity = complexity
            frame_randomness = randomness
            frame_palette = palette
            frame_seed = seed + frame  # Change seed for zoom effect
        
        # Create the frame
        img = create_wave_interference(size, frame_palette, frame_complexity, frame_randomness, frame_seed)
        images.append(img)
    
    # Create a temporary file for the GIF
    temp_file = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    
    # Save the frames as a GIF
    images[0].save(
        temp_file.name,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=frame_duration,
        loop=0
    )
    
    return temp_file.name

# Animated Color Field generator
def create_animated_color_field(size, palette, complexity, randomness, seed, frames=20, animation_type="Parameter Shift", animation_speed=0.5):
    np.random.seed(seed)
    images = []
    
    # Calculate frame duration (in ms) based on animation speed
    frame_duration = int(100 / animation_speed)  # Faster speed = shorter duration
    
    # Generate frames
    for frame in range(frames):
        # Modify parameters based on animation type
        if animation_type == "Parameter Shift":
            # Gradually shift parameters
            frame_complexity = complexity * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi))
            frame_randomness = randomness * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi + np.pi/4))
            frame_palette = palette
            frame_seed = seed
        
        elif animation_type == "Color Shift":
            # Shift colors in the palette
            frame_complexity = complexity
            frame_randomness = randomness
            
            # Convert hex colors to RGB, shift hue, and convert back to hex
            frame_palette = []
            for color in palette:
                # Convert hex to RGB
                r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                # Simple color shift (rotate RGB values)
                shift_amount = frame / frames
                r_new = int((r + shift_amount * 255) % 255)
                g_new = int((g + shift_amount * 255) % 255)
                b_new = int((b + shift_amount * 255) % 255)
                
                # Convert back to hex
                frame_palette.append(f'#{r_new:02x}{g_new:02x}{b_new:02x}')
            
            frame_seed = seed
        
        elif animation_type == "Zoom Effect":
            # Keep same parameters but change seed for a "zoom" effect
            frame_complexity = complexity
            frame_randomness = randomness
            frame_palette = palette
            frame_seed = seed + frame  # Change seed for zoom effect
        
        # Create the frame
        img = create_color_field(size, frame_palette, frame_complexity, frame_randomness, frame_seed)
        images.append(img)
    
    # Create a temporary file for the GIF
    temp_file = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    
    # Save the frames as a GIF
    images[0].save(
        temp_file.name,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=frame_duration,
        loop=0
    )
    
    return temp_file.name

# Animated Cellular Automaton generator
def create_animated_cellular_automaton(size, palette, complexity, randomness, seed, frames=20, animation_type="Parameter Shift", animation_speed=0.5):
    np.random.seed(seed)
    images = []
    
    # Calculate frame duration (in ms) based on animation speed
    frame_duration = int(100 / animation_speed)  # Faster speed = shorter duration
    
    # Generate frames
    for frame in range(frames):
        # Modify parameters based on animation type
        if animation_type == "Parameter Shift":
            # Gradually shift parameters
            frame_complexity = complexity * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi))
            frame_randomness = randomness * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi + np.pi/4))
            frame_palette = palette
            frame_seed = seed
        
        elif animation_type == "Color Shift":
            # Shift colors in the palette
            frame_complexity = complexity
            frame_randomness = randomness
            
            # Convert hex colors to RGB, shift hue, and convert back to hex
            frame_palette = []
            for color in palette:
                # Convert hex to RGB
                r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                # Simple color shift (rotate RGB values)
                shift_amount = frame / frames
                r_new = int((r + shift_amount * 255) % 255)
                g_new = int((g + shift_amount * 255) % 255)
                b_new = int((b + shift_amount * 255) % 255)
                
                # Convert back to hex
                frame_palette.append(f'#{r_new:02x}{g_new:02x}{b_new:02x}')
            
            frame_seed = seed
        
        elif animation_type == "Zoom Effect":
            # Keep same parameters but change seed for a "zoom" effect
            frame_complexity = complexity
            frame_randomness = randomness
            frame_palette = palette
            frame_seed = seed + frame  # Change seed for zoom effect
        
        # Create the frame
        img = create_cellular_automaton(size, frame_palette, frame_complexity, frame_randomness, frame_seed)
        images.append(img)
    
    # Create a temporary file for the GIF
    temp_file = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    
    # Save the frames as a GIF
    images[0].save(
        temp_file.name,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=frame_duration,
        loop=0
    )
    
    return temp_file.name

# Animated Color Field generator
def create_animated_color_field(size, palette, complexity, randomness, seed, frames=20, animation_type="Parameter Shift", animation_speed=0.5):
    images = []
    
    # Convert hex colors to RGB
    colors = [tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for color in palette]
    
    for frame in range(frames):
        # Create blank image
        img = Image.new('RGB', (size, size), color=(255, 255, 255))
        img = img.convert('RGBA')
        
        # Modify parameters based on animation type
        frame_seed = seed + frame
        np.random.seed(frame_seed)
        frame_colors = colors.copy()
        
        if animation_type == "Parameter Shift":
            # Gradually change parameters
            frame_complexity = complexity * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi))
            frame_randomness = randomness * (1 + 0.3 * np.sin(frame / frames * 2 * np.pi + np.pi/2))
        elif animation_type == "Color Shift":
            # Rotate colors
            frame_complexity = complexity
            frame_randomness = randomness
            frame_colors = colors[frame % len(colors):] + colors[:frame % len(colors)]
        else:  # Zoom Effect
            frame_complexity = complexity
            frame_randomness = randomness
            # Adjust size for zoom effect
            zoom_factor = 1 + 0.2 * np.sin(frame / frames * 2 * np.pi)
        
        # Number of shapes with animation
        num_shapes = int(frame_complexity * 20)
        
        # Draw color fields with animation
        for i in range(num_shapes):
            # Shape parameters with animation
            angle = 2 * np.pi * i / num_shapes + (frame / frames * 2 * np.pi)
            
            if animation_type == "Zoom Effect":
                # Shapes move in a circular pattern with zoom
                center_x = size // 2
                center_y = size // 2
                radius = (size // 4) * zoom_factor
                x0 = center_x + int(np.cos(angle) * radius) - size // 8
                y0 = center_y + int(np.sin(angle) * radius) - size // 8
                width = int((size // 6) * zoom_factor)
                height = int((size // 6) * zoom_factor)
            else:
                # Random shape parameters
                x0 = np.random.randint(-size//4, size)
                y0 = np.random.randint(-size//4, size)
                width = np.random.randint(size//10, size//3)
                height = np.random.randint(size//10, size//3)
            
            # Add randomness to dimensions
            if np.random.random() < frame_randomness:
                width = int(width * np.random.uniform(0.5, 1.5))
                height = int(height * np.random.uniform(0.5, 1.5))
            
            # Choose shape type (rectangle, ellipse)
            shape_type = np.random.choice(['rectangle', 'ellipse'])
            
            # Color with random alpha for blending
            color = random.choice(frame_colors)
            alpha = np.random.uniform(0.2, 0.8)
            
            # Convert to RGBA for alpha blending
            fill_color = color + (int(alpha * 255),)
            
            # Create a temporary image for this shape
            shape_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            shape_draw = ImageDraw.Draw(shape_img)
            
            # Draw the shape
            if shape_type == 'rectangle':
                # Rectangles can be rotated for more interest
                rotation_angle = np.random.uniform(0, 360) * frame_randomness
                # Draw rectangle on temporary image
                shape_draw.rectangle([x0, y0, x0+width, y0+height], fill=fill_color)
                # Rotate the shape
                shape_img = shape_img.rotate(rotation_angle + (frame / frames * 360 if animation_type == "Parameter Shift" else 0), 
                                          center=(x0+width//2, y0+height//2), resample=Image.BICUBIC)
            else:
                # Draw ellipse
                shape_draw.ellipse([x0, y0, x0+width, y0+height], fill=fill_color)
            
            # Composite the shape onto the main image
            img = Image.alpha_composite(img, shape_img)
        
        # Convert back to RGB
        img = img.convert('RGB')
        images.append(img)
    
    # Create a temporary file for the GIF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmp_file:
        # Save as animated GIF
        images[0].save(
            tmp_file.name,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=int(100 / animation_speed),
            loop=0
        )
    
    return tmp_file.name

# Animated Cellular Automaton generator
def create_animated_cellular_automaton(size, palette, complexity, randomness, seed, frames=30, animation_type="Parameter Shift", animation_speed=0.5):
    images = []
    
    # Convert palette hex colors to RGB
    colors = [tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for color in palette]
    
    # Create initial random state
    np.random.seed(seed)
    initial_grid = np.random.choice([0, 1], size=(size, size), p=[1-randomness, randomness])
    
    for frame in range(frames):
        # Modify parameters based on animation type
        frame_seed = seed
        frame_colors = colors.copy()
        
        if animation_type == "Parameter Shift":
            # Gradually change parameters
            frame_complexity = complexity * (1 + 0.2 * np.sin(frame / frames * 2 * np.pi))
            frame_randomness = randomness * (1 + 0.3 * np.sin(frame / frames * 2 * np.pi + np.pi/2))
            # Rule changes with frame
            rule_num = (int(frame_complexity * 10) + frame) % 256
        elif animation_type == "Color Shift":
            frame_complexity = complexity
            frame_randomness = randomness
            # Shift palette colors
            frame_colors = colors[frame % len(colors):] + colors[:frame % len(colors)]
            rule_num = (complexity * 10) % 256
        else:  # Zoom Effect - evolve the automaton over time
            frame_complexity = complexity
            frame_randomness = randomness
            rule_num = (complexity * 10) % 256
        
        # Define rule based on complexity and frame
        rule = [(rule_num >> i) & 1 for i in range(8)]
        
        # For zoom effect, we evolve the automaton over time
        if animation_type == "Zoom Effect":
            # Start with initial grid and evolve it frame times
            grid = initial_grid.copy()
            for _ in range(frame + 1):  # +1 to ensure we have at least one evolution
                new_grid = np.zeros_like(grid)
                for i in range(size):
                    for j in range(size):
                        # Get neighborhood (with wrapping)
                        left = grid[i, (j-1) % size]
                        center = grid[i, j]
                        right = grid[i, (j+1) % size]
                        
                        # Compute rule index
                        idx = (left << 2) | (center << 1) | right
                        
                        # Apply rule
                        new_grid[(i+1) % size, j] = rule[idx]
                grid = new_grid
        else:
            # For other animation types, create a new grid each frame
            np.random.seed(frame_seed + frame)
            grid = np.random.choice([0, 1], size=(size, size), p=[1-frame_randomness, frame_randomness])
            
            # Evolve the cellular automaton
            for _ in range(size // 2):
                new_grid = np.zeros_like(grid)
                for i in range(size):
                    for j in range(size):
                        # Get neighborhood (with wrapping)
                        left = grid[i, (j-1) % size]
                        center = grid[i, j]
                        right = grid[i, (j+1) % size]
                        
                        # Compute rule index
                        idx = (left << 2) | (center << 1) | right
                        
                        # Apply rule
                        new_grid[(i+1) % size, j] = rule[idx]
                grid = new_grid
        
        # Convert binary grid to colors
        data = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Assign colors based on patterns in the grid
        for i in range(size):
            for j in range(size):
                # Find patterns in 3x3 neighborhood
                pattern_sum = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = (i+di) % size, (j+dj) % size
                        pattern_sum += grid[ni, nj]
                
                # Map pattern to color
                color_idx = pattern_sum % len(frame_colors)
                data[i, j] = frame_colors[color_idx]
        
        # Create image from array
        img = Image.fromarray(data)
        images.append(img)
    
    # Create a temporary file for the GIF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmp_file:
        # Save as animated GIF
        images[0].save(
            tmp_file.name,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=int(100 / animation_speed),
            loop=0
        )
    
    return tmp_file.name

# Function to create and display the artwork
def generate_artwork():
    st.subheader("Your Generated Artwork")
    
    # Get the color palette
    palette = get_color_palette(color_scheme)
    
    # Handle live animation (continuously changing)
    if live_animation and not enable_animation:
        # Create a placeholder for the live animation
        live_placeholder = st.empty()
        
        # Add a stop button
        stop_button = st.button("Stop Animation")
        
        st.info("Art is continuously changing in real-time. Press 'Stop Animation' to stop.")
        
        # Display artwork details
        with st.expander("Artwork Details", expanded=False):
            st.write(f"**Style:** {art_style}")
            st.write(f"**Color Palette:** {color_scheme}")
            st.write(f"**Resolution:** {resolution}x{resolution}")
            st.write(f"**Complexity:** {complexity}")
            st.write(f"**Randomness:** {randomness}")
            st.write(f"**Base Seed:** {seed}")
        
        # Live animation loop
        frame = 0
        while not stop_button:
            # Generate a new frame with slightly modified parameters
            current_seed = seed + frame
            frame_complexity = complexity * (1 + 0.1 * np.sin(frame * 0.1))
            frame_randomness = randomness * (1 + 0.1 * np.sin(frame * 0.1 + np.pi/4))
            
            # Generate the art with the current parameters
            if art_style == "Fractal Flow":
                img = create_fractal_flow(resolution, palette, frame_complexity, frame_randomness, current_seed)
            elif art_style == "Particle System":
                img = create_particle_system(resolution, palette, frame_complexity, frame_randomness, current_seed)
            elif art_style == "Wave Interference":
                img = create_wave_interference(resolution, palette, frame_complexity, frame_randomness, current_seed)
            elif art_style == "Color Field":
                img = create_color_field(resolution, palette, frame_complexity, frame_randomness, current_seed)
            elif art_style == "Cellular Automaton":
                img = create_cellular_automaton(resolution, palette, frame_complexity, frame_randomness, current_seed)
            
            # Display the current frame
            live_placeholder.image(img, use_container_width=True)
            
            # Increment frame and add delay
            frame += 1
            time.sleep(0.2)  # Adjust for desired frame rate
            
            # Check if stop button was pressed (rerun the app)
            if st.button("Stop Animation", key=f"stop_{frame}"):
                break
            
            # Rerun to check if stop button was pressed
            if frame % 5 == 0:  # Check every 5 frames to reduce overhead
                st.experimental_rerun()
    
    # Handle GIF animation
    elif enable_animation:
        # Create a placeholder for the animation
        animation_placeholder = st.empty()
        
        # Show loading message
        with animation_placeholder.container():
            st.info(f"Generating {animation_frames} frame animation... Please wait.")
        
        # Choose the animation function based on selected style
        if art_style == "Fractal Flow":
            animation_file = create_animated_fractal_flow(
                resolution, palette, complexity, randomness, seed, 
                frames=animation_frames, animation_type=animation_type,
                animation_speed=animation_speed
            )
        elif art_style == "Particle System":
            animation_file = create_animated_particle_system(
                resolution, palette, complexity, randomness, seed,
                frames=animation_frames, animation_type=animation_type,
                animation_speed=animation_speed
            )
        elif art_style == "Wave Interference":
            animation_file = create_animated_wave_interference(
                resolution, palette, complexity, randomness, seed,
                frames=animation_frames, animation_type=animation_type,
                animation_speed=animation_speed
            )
        elif art_style == "Color Field":
            animation_file = create_animated_color_field(
                resolution, palette, complexity, randomness, seed,
                frames=animation_frames, animation_type=animation_type,
                animation_speed=animation_speed
            )
        elif art_style == "Cellular Automaton":
            animation_file = create_animated_cellular_automaton(
                resolution, palette, complexity, randomness, seed,
                frames=animation_frames, animation_type=animation_type,
                animation_speed=animation_speed
            )
        
        # Display the animation
        with open(animation_file, "rb") as f:
            animation_bytes = f.read()
        
        # Replace the loading message with the animation
        animation_placeholder.image(animation_bytes)
        
        # Download button for animation
        fn = f"animated_{art_style}_{seed}.gif"
        col1, col2, col3 = st.columns([1, 5, 1])
        with col2:
            st.download_button(
                label="Download Animated Artwork",
                data=animation_bytes,
                file_name=fn,
                mime="image/gif",
                use_container_width=True
            )
        
        # Display artwork details
        with st.expander("Artwork Details"):
            st.write(f"**Style:** {art_style}")
            st.write(f"**Color Palette:** {color_scheme}")
            st.write(f"**Resolution:** {resolution}x{resolution}")
            st.write(f"**Complexity:** {complexity}")
            st.write(f"**Randomness:** {randomness}")
            st.write(f"**Seed:** {seed}")
            st.write(f"**Animation Type:** {animation_type}")
            st.write(f"**Animation Frames:** {animation_frames}")
            st.write(f"**Animation Speed:** {animation_speed}")
            
            # Display color palette
            st.write("**Color Swatches:**")
            cols = st.columns(len(palette))
            for i, color in enumerate(palette):
                with cols[i]:
                    st.markdown(
                        f'<div style="background-color: {color}; width: 50px; height: 50px; border-radius: 5px;"></div>',
                        unsafe_allow_html=True
                    )
                    st.write(color)
        
        # Clean up the temporary file
        os.unlink(animation_file)
    
    # Handle static image generation (no animation)
    else:
        # Choose the generation function based on selected style
        if art_style == "Fractal Flow":
            img = create_fractal_flow(resolution, palette, complexity, randomness, seed)
        elif art_style == "Particle System":
            img = create_particle_system(resolution, palette, complexity, randomness, seed)
        elif art_style == "Wave Interference":
            img = create_wave_interference(resolution, palette, complexity, randomness, seed)
        elif art_style == "Color Field":
            img = create_color_field(resolution, palette, complexity, randomness, seed)
        elif art_style == "Cellular Automaton":
            img = create_cellular_automaton(resolution, palette, complexity, randomness, seed)
        
        # Display the image
        col1, col2, col3 = st.columns([1, 5, 1])
        with col2:
            st.image(img, use_container_width=True)
        
        # Create a buffer to save the image
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        # Download button
        fn = f"abstract_art_{art_style}_{seed}.png"
        col4, col5, col6 = st.columns([1, 5, 1])
        with col5:
            st.download_button(
                label="Download Artwork",
                data=byte_im,
                file_name=fn,
                mime="image/png",
                use_container_width=True
            )
        
        # Display artwork details
        with st.expander("Artwork Details"):
            st.write(f"**Style:** {art_style}")
            st.write(f"**Color Palette:** {color_scheme}")
            st.write(f"**Resolution:** {resolution}x{resolution}")
            st.write(f"**Complexity:** {complexity}")
            st.write(f"**Randomness:** {randomness}")
            st.write(f"**Seed:** {seed}")
            
            # Display color palette
            st.write("**Color Swatches:**")
            cols = st.columns(len(palette))
            for i, color in enumerate(palette):
                with cols[i]:
                    st.markdown(
                        f'<div style="background-color: {color}; width: 50px; height: 50px; border-radius: 5px;"></div>',
                        unsafe_allow_html=True
                    )
                    st.write(color)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Generate art when button is clicked
    if generate_button:
        generate_artwork()
    else:
        # Show welcome message and instructions if no art has been generated yet
        st.write("### Welcome to the Abstract Art Generator!")
        st.write("""
        This app creates unique generative artworks based on mathematical algorithms.
        
        To create your own abstract artwork:
        1. Choose an art style from the sidebar
        2. Select a color scheme
        3. Adjust the resolution, complexity, and randomness
        4. Optional: Set a specific seed or use a random one
        5. Click the "Generate Artwork" button
        
        Each style creates different types of abstract art:
        - **Fractal Flow**: Creates flowing patterns based on fractal mathematics
        - **Particle System**: Generates compositions of particles and connections
        - **Wave Interference**: Simulates overlapping wave patterns
        - **Color Field**: Creates abstract compositions with overlapping color shapes
        - **Cellular Automaton**: Produces complex patterns from simple cellular rules
        """)
        
        # Show sample preview images
        st.write("### Style Examples:")
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/streamlit-logo-secondary-colormark-darktext.png", 
                 caption="Sample image - your generated art will appear here",
                 use_container_width=True)

with col2:
    # Information and tips
    with st.expander("Tips & Tricks"):
        st.write("""
        - Try different combinations of styles and color schemes
        - Higher complexity creates more detailed artwork but takes longer to generate
        - Increase randomness for more unpredictable results
        - Use the same seed to recreate a specific artwork
        - Download your creations to save them
        """)
    
    # About section
    with st.expander("About"):
        st.write("""
        This Streamlit app demonstrates how to create generative art using various algorithms.
        
        Created with ❤️ using Streamlit, Matplotlib, NumPy, and Pillow.
        """)

# Footer
st.markdown("""
---
<p style="text-align: center; color: #aaaaaa; font-size: 12px;">
Abstract Art Generator • Powered by Streamlit
</p>
""", unsafe_allow_html=True)