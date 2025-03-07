#сравниваем два изображения: оригинальные разметку и аннотацию и предсказанные нашей моделью с заданными параметрами


from PIL import Image
import matplotlib.pyplot as plt
from detect import detect

img_path = '/content/BCCD_Dataset/BCCD/JPEGImages/' #путь к оригинальному файлу
annotated_path = '/content/imagesBox/' #путь к файлу с разметкой
file_name = 'BloodImage_00008.jpg' #название файла


annotated_image = Image.open(annotated_path+file_name).convert('RGB')


img = Image.open(img_path+file_name).convert('RGB')
predicted_image = detect(img, min_score=0.2, max_overlap=0.5, top_k=200)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Оригинальное изображение
axs[0].imshow(annotated_image)
axs[0].set_title("Original Image")
axs[0].axis("off")

# Аннотированное изображение
axs[1].imshow(predicted_image)
axs[1].set_title("predicted_image")
axs[1].axis("off")

# Показать оба изображения
plt.show()

