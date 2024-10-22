
import os
from PIL import Image



source_dir = '/Users/dimitrichrysafis/PycharmProjects/setting/cards'
destination_dir = '/Users/dimitrichrysafis/PycharmProjects/setting/processedcardscolorfilter'

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

def remove_white_background(image):
    img = image.convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    return img

def apply_color_filter(image, color):
    img = image.convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        r, g, b, a = item

        if color == 'red' and r > g and r > b:
            new_data.append(item)
        elif color == 'green' and g > r and g > b:
            new_data.append(item)
        elif color == 'purple' and b > r and b > g:
            new_data.append(item)
        else:
            new_data.append((0, 0, 0, 0))

    img.putdata(new_data)
    return img

def crop_borders(image, border_size):
    width, height = image.size
    cropped_image = image.crop((border_size, border_size, width - border_size, height - border_size))
    return cropped_image

def compare_images(original, filtered):
    original_data = original.getdata()
    filtered_data = filtered.getdata()

    original_non_white_pixels = 0
    filtered_non_transparent_pixels = 0

    for orig_pixel, filt_pixel in zip(original_data, filtered_data):
        if orig_pixel[3] > 0:
            original_non_white_pixels += 1
        if filt_pixel[3] > 0:
            filtered_non_transparent_pixels += 1

    if original_non_white_pixels == 0:
        return 0
    return filtered_non_transparent_pixels / original_non_white_pixels

def process_images(border_trim=10):
    for image_name in os.listdir(source_dir):
        if image_name.endswith(".png"):
            image_path = os.path.join(source_dir, image_name)

            try:
                img = Image.open(image_path)
                img = remove_white_background(img)

                img_cropped = crop_borders(img, border_trim)

                color_proportions = {}

                for color in ['red', 'green', 'purple']:
                    filtered_img = apply_color_filter(img_cropped, color)
                    filtered_image_path = os.path.join(destination_dir, f"{image_name.split('.')[0]}_{color}.png")

                    filtered_img.save(filtered_image_path, "PNG")

                    similarity = compare_images(img_cropped, filtered_img)
                    color_proportions[color] = similarity

                most_prominent_color = max(color_proportions, key=color_proportions.get)

                for color in ['red', 'green', 'purple']:
                    if color != most_prominent_color:
                        os.remove(os.path.join(destination_dir, f"{image_name.split('.')[0]}_{color}.png"))

                print(f"{image_name}: Most prominent color is {most_prominent_color}")

            except Exception as e:
                print(f"Error processing {image_name}: {e}")

process_images()
