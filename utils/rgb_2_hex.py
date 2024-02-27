import json
import matplotlib.colors as colors

# Load the color table
with open('color_table.json', 'r') as f:
    color_table = json.load(f)

# Convert the RGB values to hexadecimal
for key in color_table:
    rgb_values = color_table[key]
    hex_color = colors.rgb2hex(rgb_values)
    color_table[key] = hex_color

# Save the color table with hexadecimal values
with open('color_table_hex.json', 'w') as f:
    json.dump(color_table, f)