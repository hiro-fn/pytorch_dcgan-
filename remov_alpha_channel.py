from PIL import Image
from glob import glob

def remove_alpha_channel(image_path):
    image = Image.open(image_path)
    resule_image = []
    count = 0
    if image == "RGBA":
        result_image = image.convert("RGBA")
    else:
        result_image = image

    return result_image

image_list = glob('D:\project\dcgan2\dataset\\keiko\*')
print(image_list)

print("Begin Process")
jpeg_not_alpha_channel = remove_alpha_channel_image_list = list(map(lambda f: remove_alpha_channel(f),
                                                                image_list ))
print("End Process")

for count, f in enumerate(jpeg_not_alpha_channel):
    print('=' * 10)
    print(count)
    print(f)
    try:
        f.save(f'D:\project\dcgan2\dataset\\alpha\{count}.jpg', quality=100, optimize=True)
    except Exception as e:
        print(str(e))
    print()

    print("END SAVE\n")
