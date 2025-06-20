import base64

class ImageToBase64:
    def __init__(self):
        pass

    def image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

if __name__ == "__main__":
    image_path = r"C:\Users\ASUS\Desktop\workspace\Automated_Visualization\backend\problems\object_detection_in_image\data\000000001257.jpg"
    print(ImageToBase64().image_to_base64(image_path))
    with open("base64_image.txt", "w") as f:
        f.write(ImageToBase64().image_to_base64(image_path))